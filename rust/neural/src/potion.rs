//! The potion static embedder: the model's `tokenizer.json` through the
//! WordPiece tokenizer, the rows of its `model.safetensors` gathered, mean
//! pooled and l2 normalized by the native backend. The recipe and the
//! `model_hash` are the reference script's (`potiontab.py`), field for
//! field, so a corpus's manifest gate accepts the vector exactly as it
//! accepts the script's.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::backend::Native;
use crate::model::StaticModel;
use crate::tokenizer::Tokenizer;
use crate::{InferError, Result, SafeTensors};

const MODEL_ID: &str = "minishlab/potion-base-8M";
const POTION_VERSION: &str = "1";
/// the files whose bytes the model_hash pins, in `RELEVANT_FILES` order.
const RELEVANT_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

pub struct Potion {
    tok: Tokenizer,
    model: StaticModel<Native>,
    model_hash: String,
}

impl std::fmt::Debug for Potion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potion")
            .field("model", &self.model)
            .field("model_hash", &self.model_hash)
            .finish_non_exhaustive()
    }
}

impl Potion {
    /// Load the table dir: `tokenizer.json`, `model.safetensors` and the
    /// `config.json` the fingerprint pins.
    pub fn open(dir: &Path) -> Result<Self> {
        let tok = Tokenizer::from_file(&dir.join("tokenizer.json"))?;
        let path = dir.join("model.safetensors");
        let st = SafeTensors::open(&path)?;
        let dim = st
            .shape("embeddings")
            .and_then(|s| s.get(1).copied())
            .ok_or_else(|| InferError::SafeTensors {
                path: path.display().to_string(),
                reason: "no [rows, dim] embeddings tensor".into(),
            })?;
        let table = st.tensor_f32("embeddings")?;
        let unk = tok.token_to_id("[UNK]").unwrap_or(1);
        let model = StaticModel::new(Native::detect(), table, dim, unk)?;
        let model_hash = model_hash(dir, dim)?;
        Ok(Self {
            tok,
            model,
            model_hash,
        })
    }

    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    pub fn dim(&self) -> usize {
        self.model.dim()
    }

    /// The query's vector: the token rows' mean, unit norm; an empty text
    /// embeds as the unk row, like the script.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tok.encode(text);
        self.model.embed_ids(&ids)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| InferError::SafeTensors {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// `sha256:` of the canonical json fingerprint (keys sorted, no spaces), as
/// `PotionEmbedder.fingerprint` + `model_hash` write it.
fn model_hash(dir: &Path, dim: usize) -> Result<String> {
    if dim == 0 {
        return Err(InferError::Shape("potion table has no columns".into()));
    }
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
