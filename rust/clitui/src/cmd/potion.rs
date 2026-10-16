//! The potion query embedder in process (#205): tokenize with the vendored
//! `tokenizer.json` (no special tokens), gather the rows of the vendored
//! `model.safetensors`, mean pool, l2 normalize. The recipe and the
//! `model_hash` are `python/urna/embed/potiontab.py`'s, field for field, so
//! the manifest gate accepts the vector exactly as it accepts the script's.
//! The table loads once per process (~30 MB, ~90 ms) and every query after
//! is a table lookup.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "minishlab/potion-base-8M";
const POTION_VERSION: &str = "1";
/// the files whose bytes the model_hash pins, in `RELEVANT_FILES` order.
const RELEVANT_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

pub struct Potion {
    tok: Tokenizer,
    table: Vec<f32>,
    dim: usize,
    unk: u32,
    model_hash: String,
}

impl std::fmt::Debug for Potion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potion")
            .field("dim", &self.dim)
            .field("rows", &(self.table.len() / self.dim.max(1)))
            .field("model_hash", &self.model_hash)
            .finish_non_exhaustive()
    }
}

/// The table dir next to the potion script: `<pkg>/embed/potionqry.py` ->
/// `<pkg>/model/potionb8m`.
pub fn table_dir(script: &Path) -> Option<PathBuf> {
    let dir = script.parent()?.parent()?.join("model").join("potionb8m");
    dir.join("model.safetensors").is_file().then_some(dir)
}

/// The table dir and its loaded embedder, the one this process keeps.
type Loaded = Option<(PathBuf, Arc<Potion>)>;

/// The loaded embedder for `dir`, shared by every query of the process.
pub fn shared(dir: &Path) -> Result<Arc<Potion>> {
    static CACHE: OnceLock<Mutex<Loaded>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut slot = cell.lock().map_err(|_| anyhow!("potion cache poisoned"))?;
    if let Some((d, p)) = slot.as_ref()
        && d == dir
    {
        return Ok(Arc::clone(p));
    }
    let p = Arc::new(Potion::open(dir)?);
    *slot = Some((dir.to_path_buf(), Arc::clone(&p)));
    Ok(p)
}

impl Potion {
    pub fn open(dir: &Path) -> Result<Self> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("{}: {e}", dir.join("tokenizer.json").display()))?;
        let (table, dim) = read_table(&dir.join("model.safetensors"))?;
        let unk = tok.token_to_id("[UNK]").unwrap_or(1);
        let model_hash = model_hash(dir, dim)?;
        Ok(Self {
            tok,
            table,
            dim,
            unk,
            model_hash,
        })
    }

    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        // an empty text embeds as the unk row, like the script.
        let ids: &[u32] = if enc.get_ids().is_empty() {
            std::slice::from_ref(&self.unk)
        } else {
            enc.get_ids()
        };
        let mut v = vec![0f32; self.dim];
        for &id in ids {
            let start = id as usize * self.dim;
            let row = self
                .table
                .get(start..start + self.dim)
                .ok_or_else(|| anyhow!("token id {id} past the table"))?;
            v.iter_mut().zip(row).for_each(|(a, b)| *a += b);
        }
        let n = ids.len() as f32;
        v.iter_mut().for_each(|a| *a /= n);
        let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|a| *a /= norm);
        }
        Ok(v)
    }
}

/// The single F32 `embeddings` tensor: 8-byte header length, json header,
/// little-endian payload.
fn read_table(path: &Path) -> Result<(Vec<f32>, usize)> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    if bytes.starts_with(b"version https://git-lfs") {
        bail!(
            "{} is a git-lfs pointer; run `git lfs pull`",
            path.display()
        );
    }
    let head: [u8; 8] = bytes
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| anyhow!("{}: truncated", path.display()))?;
    let n = u64::from_le_bytes(head) as usize;
    let header: serde_json::Value = serde_json::from_slice(
        bytes
            .get(8..8 + n)
            .ok_or_else(|| anyhow!("{}: truncated header", path.display()))?,
    )?;
    let t = &header["embeddings"];
    if t["dtype"] != "F32" {
        bail!("{}: embeddings is not F32", path.display());
    }
    let num = |v: &serde_json::Value| {
        v.as_u64()
            .map(|x| x as usize)
            .ok_or_else(|| anyhow!("{}: bad header", path.display()))
    };
    let (dim, a, b) = (
        num(&t["shape"][1])?,
        num(&t["data_offsets"][0])?,
        num(&t["data_offsets"][1])?,
    );
    let data = bytes
        .get(8 + n + a..8 + n + b)
        .ok_or_else(|| anyhow!("{}: data out of range", path.display()))?;
    let table = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok((table, dim))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// `sha256:` of the canonical json fingerprint (keys sorted, no spaces), as
/// `PotionEmbedder.fingerprint` + `model_hash` write it.
fn model_hash(dir: &Path, dim: usize) -> Result<String> {
    let mut files = Sha256::new();
    for rel in RELEVANT_FILES {
        let p = dir.join(rel);
        if !p.is_file() {
            continue;
        }
        files.update(rel.as_bytes());
        files.update(b"\0");
        files.update(sha256_file(&p)?.as_bytes());
        files.update(b"\0");
    }
    let fp = serde_json::json!({
        "add_special_tokens": false,
        "dtype": "float32-stable",
        "embedder": MODEL_ID,
        "embedding_dim": dim,
        "files_hash": format!("sha256:{}", hex::encode(files.finalize())),
        "normalize": "l2",
        "pooling": "mean",
        "tokenizer": "bert-wordpiece-lower",
        "tokenizer_hash": format!("sha256:{}", sha256_file(&dir.join("tokenizer.json"))?),
        "version": POTION_VERSION,
    });
    Ok(format!(
        "sha256:{}",
        hex::encode(Sha256::digest(fp.to_string().as_bytes()))
    ))
}
