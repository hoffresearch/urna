//! The ONE copy of the query-embedder spawn protocol and the three-layer
//! model gate (name -> dim -> model_hash). `search-text`, `ask` and
//! `retrieve` all come through here.
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

/// Find the sentence-transformers embedder (`urna/embed/searchtxt.py`, the
/// path `search-text` keeps as its default): repo layout, then the data
/// roots, where the payload lays it down beside the potion one.
pub fn default_embedder_path() -> PathBuf {
    installed_script("searchtxt.py")
}

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
/// and `retrieve`; `search-text` uses the pieces directly (it keeps
/// `--skip-model-hash-check`).
pub fn embed_and_search(
    runtime: &MmapUrnaFile,
    query: &str,
    k: i32,
    candidates: Option<usize>,
    embedder: Option<PathBuf>,
    model_path: Option<PathBuf>,
) -> Result<SearchResult> {
    let info: serde_json::Value = serde_json::from_str(&runtime.inspect_json()?)?;
    let model = manifest_str(&info, "embedding_model")?;
    let declared_dim = info["manifest"]["embedding_dim"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("manifest.embedding_dim missing"))?
        as usize;
    let declared_model_hash = manifest_str(&info, "model_hash")?;

    // route by manifest model: potion corpora keep the potion script; any
    // registry model goes through the generic embedder, with --mrl-dim when
    // the default space was truncated (full_dim recorded).
    let is_potion = model.starts_with("minishlab/potion");
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
    if info["manifest"]["full_dim"].as_u64().is_some() {
        extra.push("--mrl-dim".into());
        extra.push(declared_dim.to_string());
    }
    let payload = spawn_embedder(&embedder, model_path.as_ref(), &extra, &model, query)?;
    validate_gate(&payload, &model, declared_dim, &declared_model_hash, false)?;

    // route by capability (bm25 -> hybrid, hnsw -> ann, else exact), not by
    // the declared `index_type`: a `hybrid` preset file declares "hnsw" and
    // carries its bm25 section as a capability.
    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    Ok(runtime.search_routed(&payload.vector, Some(query), k, cand)?)
}

fn manifest_str(info: &serde_json::Value, key: &str) -> Result<String> {
    info["manifest"][key]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("manifest.{} missing", key))
}
