//! The ONE copy of the query-embedder spawn protocol and the three-layer
//! model gate (name -> dim -> model_hash). `search-text`, `ask` and
//! `retrieve` all come through here; before this module the gate lived
//! twice and could drift.
//!
//! Routing: a manifest whose `embedding_model` starts with the potion
//! prefix keeps the offline potion script (status quo, byte-for-byte);
//! any other model routes to `python/forge/embed_query_model.py`, the
//! registry-backed embedder, passing `--mrl-dim` when the manifest
//! records a truncated default space (`full_dim` present).

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command as ProcCommand;

use urna_runtime::{MmapUrnaFile, SearchResult};

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

/// Walk up from the current dir to find python/embed_query.py (the legacy
/// sentence-transformers path that `search-text` keeps as its default).
pub fn default_embedder_path() -> PathBuf {
    repo_script(&["python", "embed_query.py"])
        .unwrap_or_else(|| PathBuf::from("python/embed_query.py"))
}

/// Find the OFFLINE potion embedder (`python/forge/embed_query_potion.py`):
/// repo layout, then the installed data dir, then `<exe>/../share` (the
/// installer/tarball layouts, issue #75).
pub fn default_potion_embedder_path() -> PathBuf {
    installed_script("embed_query_potion.py")
}

/// Find the registry-backed embedder (`python/forge/embed_query_model.py`),
/// same resolution ladder as the potion script.
pub fn default_registry_embedder_path() -> PathBuf {
    installed_script("embed_query_model.py")
}

fn repo_script(rel: &[&str]) -> Option<PathBuf> {
    let rel: PathBuf = rel.iter().collect();
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
        let c = base.join(&rel);
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
    if repo.join("python").is_dir() {
        Some(repo.to_path_buf())
    } else {
        None
    }
}

fn installed_script(name: &str) -> PathBuf {
    installed_script_in("forge", name)
}

/// One resolution ladder for every shipped python script: repo layout
/// (`python/<subdir>/<name>` walking up from cwd, then beside the exe),
/// then the installed data dir, then `<exe>/../share` (issue #75 layouts).
pub(crate) fn installed_script_in(subdir: &str, name: &str) -> PathBuf {
    if let Some(p) = repo_script(&["python", subdir, name]) {
        return p;
    }
    let data_home = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local").join("share"))
        });
    let exe_share = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
        .map(|bin| bin.join("..").join("share"));
    for base in [data_home, exe_share].into_iter().flatten() {
        let c = base.join("urna").join(subdir).join(name);
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("python").join(subdir).join(name)
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
        anyhow::bail!(
            "embedder script not found: {} (override with --embedder)",
            embedder.display()
        );
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
        anyhow::bail!(
            "embedder failed (status={}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
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
    if payload.embedding_model != model {
        anyhow::bail!(
            "model name mismatch: manifest={}, embedder reports={}",
            model,
            payload.embedding_model
        );
    }
    if payload.embedding_dim != declared_dim || payload.vector.len() != declared_dim {
        anyhow::bail!(
            "dim mismatch: manifest={}, embedder dim={}, vector len={}",
            declared_dim,
            payload.embedding_dim,
            payload.vector.len()
        );
    }
    if skip_model_hash_check {
        return Ok(());
    }
    if declared_model_hash == PLACEHOLDER_MODEL_HASH {
        anyhow::bail!(
            "manifest carries the legacy placeholder model_hash ({}). Rebuild \
             this corpus with a real fingerprint, or pass --skip-model-hash-check \
             to proceed at your own risk.",
            PLACEHOLDER_MODEL_HASH
        );
    }
    if payload.model_hash != declared_model_hash {
        anyhow::bail!(
            "model_hash mismatch: corpus was built with {}, embedder reports {}\n\
             fingerprint reported by embedder: {}\n\
             hint: --model-path PATH to point at the exact snapshot, or rebuild \
             the corpus with the model you intend to use.",
            declared_model_hash,
            payload.model_hash,
            payload.fingerprint
        );
    }
    Ok(())
}

/// Embed `query` offline, gate it against the manifest, route by declared
/// capability, search. Shared by `ask` and `retrieve`; `search-text` uses
/// the pieces directly (it keeps `--skip-model-hash-check`).
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

    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let result = match runtime.declared_index_type() {
        "hnsw" => runtime.search_ann(&payload.vector, k, cand)?,
        "hybrid" => runtime.search_hybrid(&payload.vector, query, k, cand)?,
        "graph" => runtime.search_graph(&payload.vector, k, 1, cand)?,
        _ => runtime.search(&payload.vector, k)?,
    };
    Ok(result)
}

fn manifest_str(info: &serde_json::Value, key: &str) -> Result<String> {
    info["manifest"][key]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("manifest.{} missing", key))
}
