//! The ONE copy of the query-embedder spawn protocol and the three-layer
//! model gate (name -> dim -> model_hash). `search-text`, `ask`,
//! `retrieve` and the tui all come through `embed_and_route`.
//!
//! Routing: a manifest whose `embedding_model` starts with the potion
//! prefix keeps the offline potion script; any other model routes to
//! `urna/embed/presetqry.py`, the registry-backed embedder, passing
//! `--mrl-dim` when the manifest records a truncated default space
//! (`full_dim` present).

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command as ProcCommand;

use urna_engine::{MmapUrnaFile, SearchResult};

use super::embed_failure::EmbedFailure;

/// Output schema shared by every query embedder script: the compact
/// `model_hash` is the source of truth for the gate; `fingerprint` is
/// diagnostic only.
#[derive(serde::Deserialize)]
pub struct EmbedderOutput {
    pub model_hash: String,
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub fingerprint: serde_json::Value,
}

pub const PLACEHOLDER_MODEL_HASH: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

/// Find the OFFLINE potion embedder (`urna/embed/potionqry.py`):
/// repo layout, then the installed data dir, then `<exe>/../share` (the
/// installer/tarball layouts, issue #75).
pub fn default_potion_embedder_path() -> PathBuf {
    installed_script("potionqry.py")
}

/// Find the registry-backed embedder (`urna/embed/presetqry.py`),
/// same resolution ladder as the potion script.
pub fn default_registry_embedder_path() -> PathBuf {
    installed_script("presetqry.py")
}

fn repo_script(rel: &Path) -> Option<PathBuf> {
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd.clone());
        bases.push(cwd.join(".."));
    }
    // a dev-built binary lives at <repo>/target/release/urna; resolve against
    // its own checkout too, so the binary works when invoked from another
    // repository (e.g. as a git submodule's toolchain).
    if let Some(repo) = exe_repo_root() {
        bases.push(repo);
    }
    for base in bases {
        let c = base.join(rel);
        if c.exists() {
            return Some(c);
        }
    }
    None
}

/// <repo> for a dev-built binary at <repo>/target/<profile>/urna.
pub(crate) fn exe_repo_root() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let repo = exe.parent()?.parent()?.parent()?;
    if repo.join(package()).is_dir() {
        Some(repo.to_path_buf())
    } else {
        None
    }
}

fn installed_script(name: &str) -> PathBuf {
    installed_script_in(&["embed", name])
}

/// The `urna` python package in a checkout (`rust/bridge/python/urna`).
fn package() -> PathBuf {
    ["rust", "bridge", "python", "urna"].iter().collect()
}

/// One resolution ladder for every shipped python script, `rel` being its
/// path in the `urna` package: repo layout (`rust/bridge/python/urna/<rel>`
/// walking up from cwd, then beside the exe), then the payload's
/// `<root>/urna/python/urna/<rel>` for every data root in
/// `paths::data_roots` (issue #75 layouts).
pub(crate) fn installed_script_in(rel: &[&str]) -> PathBuf {
    let rel: PathBuf = rel.iter().collect();
    if let Some(p) = repo_script(&package().join(&rel)) {
        return p;
    }
    for base in super::paths::data_roots() {
        let c = super::payload::package_in(&base).join(&rel);
        if c.exists() {
            return c;
        }
    }
    package().join(rel)
}

/// Spawn the embedder script and parse its one-line JSON payload.
/// argv: `<interp> <script> [--model-path P] [extra...] <model> <query>`.
pub fn spawn_embedder(
    embedder: &PathBuf,
    model_path: Option<&PathBuf>,
    extra_args: &[String],
    model: &str,
    query: &str,
) -> Result<EmbedderOutput> {
    if !embedder.exists() {
        return Err(EmbedFailure::ScriptMissing {
            script: embedder.clone(),
        }
        .into());
    }
    let interpreter = super::pyenv::resolve_interpreter();
    if !super::pyenv::may_run(&interpreter) {
        return Err(EmbedFailure::PythonMissing {
            python: interpreter,
        }
        .into());
    }
    let mut cmd = ProcCommand::new(&interpreter);
    cmd.arg(embedder);
    if let Some(p) = model_path {
        cmd.arg("--model-path").arg(p);
    }
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.arg(model).arg(query);
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn embedder: {} ({})", e, embedder.display()))?;
    if !out.status.success() {
        // a script inside an installed payload that lacks files fails for
        // that reason, whatever python printed (a missing module, a missing
        // tokenizer): name the files and the repair.
        if let Some(home) = super::payload::home_of(embedder) {
            let missing = super::payload::missing(&home);
            if !missing.is_empty() {
                return Err(EmbedFailure::PayloadIncomplete {
                    home: Some(home),
                    missing: missing.into_iter().map(String::from).collect(),
                }
                .into());
            }
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        let status = out.status.to_string();
        return Err(EmbedFailure::from_embedder(&status, &stderr, model, &interpreter).into());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| {
        anyhow::anyhow!(
            "invalid embedder output: {} (stdout={:?})",
            e,
            String::from_utf8_lossy(&out.stdout)
        )
    })
}

/// The three-layer gate: name (layer 1), dim (layer 2), model_hash
/// (layer 3, skippable only for legacy placeholder corpora).
pub fn validate_gate(
    payload: &EmbedderOutput,
    model: &str,
    declared_dim: usize,
    declared_model_hash: &str,
    skip_model_hash_check: bool,
) -> Result<()> {
    let incompatible = |detail: String| EmbedFailure::ModelIncompatible {
        model: model.to_string(),
        detail,
    };
    if payload.embedding_model != model {
        return Err(incompatible(format!(
            "model name mismatch: manifest={}, embedder reports={}",
            model, payload.embedding_model
        ))
        .into());
    }
    if payload.embedding_dim != declared_dim || payload.vector.len() != declared_dim {
        return Err(incompatible(format!(
            "dim mismatch: manifest={}, embedder dim={}, vector len={}",
            declared_dim,
            payload.embedding_dim,
            payload.vector.len()
        ))
        .into());
    }
    if declared_model_hash == PLACEHOLDER_MODEL_HASH {
        // the one case the flag covers: a legacy corpus with no fingerprint
        // to compare against. the caller accepts the risk explicitly.
        if skip_model_hash_check {
            return Ok(());
        }
        anyhow::bail!(
            "manifest carries the legacy placeholder model_hash ({}). Rebuild \
             this corpus with a real fingerprint, or pass --skip-model-hash-check \
             to proceed at your own risk.",
            PLACEHOLDER_MODEL_HASH
        );
    }
    if payload.model_hash != declared_model_hash {
        // a real fingerprint that disagrees is never skippable: the hits
        // would be cosine-valid and wrong.
        return Err(EmbedFailure::HashMismatch {
            corpus: declared_model_hash.to_string(),
            embedder: payload.model_hash.clone(),
            fingerprint: payload.fingerprint.to_string(),
        }
        .into());
    }
    Ok(())
}

/// Embed `query` offline, gate it against the manifest, route by what the
/// file carries (`MmapUrnaFile::search_routed`), search. Shared by `ask`
/// and `retrieve`.
pub fn embed_and_search(
    runtime: &MmapUrnaFile,
    query: &str,
    k: i32,
    candidates: Option<usize>,
    embedder: Option<PathBuf>,
    model_path: Option<PathBuf>,
) -> Result<SearchResult> {
    embed_and_search_where(runtime, query, k, candidates, embedder, model_path, None)
}

/// `embed_and_search` over the chunks `keep` accepts: the bitset a filter
/// resolved is applied before scoring on every route.
pub fn embed_and_search_where(
    runtime: &MmapUrnaFile,
    query: &str,
    k: i32,
    candidates: Option<usize>,
    embedder: Option<PathBuf>,
    model_path: Option<PathBuf>,
    keep: Option<&urna_engine::scalars::Bitset>,
) -> Result<SearchResult> {
    embed_and_route(
        runtime, query, k, candidates, embedder, model_path, keep, false,
    )
}

/// The one embed, gate and route every query command runs; only
/// `search-text` may pass `skip_model_hash_check`, for a legacy corpus
/// that carries the placeholder model_hash.
#[allow(clippy::too_many_arguments)]
pub fn embed_and_route(
    runtime: &MmapUrnaFile,
    query: &str,
    k: i32,
    candidates: Option<usize>,
    embedder: Option<PathBuf>,
    model_path: Option<PathBuf>,
    keep: Option<&urna_engine::scalars::Bitset>,
    skip_model_hash_check: bool,
) -> Result<SearchResult> {
    // the manifest fields the open already parsed: no second pass over the
    // file (inspect_json re-verifies every checksum, ~1 s on a 1 GB corpus).
    let model = runtime.embedding_model().to_string();
    let declared_dim = runtime.embedding_dim();
    let declared_model_hash = runtime.model_hash().to_string();

    // route by manifest model: potion corpora keep the potion script; any
    // registry model goes through the generic embedder, with --mrl-dim when
    // the default space was truncated (full_dim recorded).
    let is_potion = urna_neural::Registry::for_model(&model).is_some();
    let truncated = runtime.full_dim().is_some();
    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let unchecked = hash_skipped(skip_model_hash_check, &declared_model_hash);

    // a potion corpus queried with the default embedder embeds in process
    // (#205): same recipe, same model_hash, no interpreter. the script stays
    // for `--embedder`, `--model-path`, mrl-truncated corpora and a payload
    // without the table.
    if is_potion && embedder.is_none() && model_path.is_none() && !truncated {
        let table = super::potion::table_dir(&default_potion_embedder_path());
        if let Some(dir) = table {
            if unchecked {
                say_unchecked(&model, &format!("the potion table in {}", dir.display()));
            }
            let p = super::potion::shared(&dir)?;
            let payload = EmbedderOutput {
                model_hash: p.model_hash().to_string(),
                // the table was chosen by the manifest's model name.
                embedding_model: model.clone(),
                embedding_dim: p.dim(),
                vector: p.embed(query)?,
                fingerprint: serde_json::Value::Null,
            };
            validate_gate(
                &payload,
                &model,
                declared_dim,
                &declared_model_hash,
                skip_model_hash_check,
            )?;
            return Ok(runtime.search_routed_where(&payload.vector, Some(query), k, cand, keep)?);
        }
    }

    let embedder = embedder.unwrap_or_else(|| {
        if is_potion {
            default_potion_embedder_path()
        } else {
            default_registry_embedder_path()
        }
    });
    // both embedders take --mrl-dim, so a corpus whose default space was
    // truncated (full_dim recorded) is queryable regardless of the model.
    let mut extra: Vec<String> = Vec::new();
    if truncated {
        extra.push("--mrl-dim".into());
        extra.push(declared_dim.to_string());
    }
    if unchecked {
        say_unchecked(&model, &embedder.display().to_string());
    }
    let payload = spawn_embedder(&embedder, model_path.as_ref(), &extra, &model, query)?;
    validate_gate(
        &payload,
        &model,
        declared_dim,
        &declared_model_hash,
        skip_model_hash_check,
    )?;

    // route by capability (bm25 -> hybrid, hnsw -> ann, else exact), not by
    // the declared `index_type`: a `hybrid` preset file declares "hnsw" and
    // carries its bm25 section as a capability.
    Ok(runtime.search_routed_where(&payload.vector, Some(query), k, cand, keep)?)
}

/// The flag skips the gate only on the legacy placeholder (`validate_gate`);
/// a real fingerprint is checked whatever the flag says.
fn hash_skipped(skip_model_hash_check: bool, declared_model_hash: &str) -> bool {
    skip_model_hash_check && declared_model_hash == PLACEHOLDER_MODEL_HASH
}

/// With the model_hash gate off, nothing proves the query was embedded the
/// way the corpus was: the line names what embedded it, so a ranking that
/// looks off can be traced to the embedder.
fn say_unchecked(model: &str, via: &str) {
    eprintln!("[urna] model_hash not checked: embedding the query with {model} via {via}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_skips_only_the_placeholder() {
        assert!(hash_skipped(true, PLACEHOLDER_MODEL_HASH));
        assert!(!hash_skipped(true, "sha256:real"));
        assert!(!hash_skipped(false, PLACEHOLDER_MODEL_HASH));
    }
}
