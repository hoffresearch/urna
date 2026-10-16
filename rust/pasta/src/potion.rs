//! The potion-base-8M query embedder in process: tokenize without special
//! tokens, gather the table rows, mean pool, l2 normalize. The same recipe
//! and the same `model_hash` as urna's `potiontab.py`, so the file's gate
//! accepts the vector.

use std::path::Path;

use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::Error;

const MODEL_ID: &str = "minishlab/potion-base-8M";
const POTION_VERSION: &str = "1";

pub struct Potion {
    tok: Tokenizer,
    table: Vec<f32>,
    dim: usize,
    unk: u32,
    model_hash: String,
}

impl Potion {
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;
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

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;
        let ids: Vec<u32> = if enc.get_ids().is_empty() {
            vec![self.unk]
        } else {
            enc.get_ids().to_vec()
        };
        let mut v = vec![0f32; self.dim];
        for id in &ids {
            let row = &self.table[*id as usize * self.dim..(*id as usize + 1) * self.dim];
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

/// The single F32 `embeddings` tensor of a safetensors file.
fn read_table(path: &Path) -> Result<(Vec<f32>, usize), Error> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    let bad = |why: &str| Error::Model(format!("{}: {why}", path.display()));
    let n = u64::from_le_bytes(
        bytes
            .get(..8)
            .ok_or_else(|| bad("truncated"))?
            .try_into()
            .map_err(|_| bad("header"))?,
    ) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..8 + n).ok_or_else(|| bad("truncated header"))?)
            .map_err(|_| bad("header json"))?;
    let t = &header["embeddings"];
    if t["dtype"] != "F32" {
        return Err(bad("embeddings is not F32"));
    }
    let dim = t["shape"][1].as_u64().ok_or_else(|| bad("shape"))? as usize;
    let (a, b) = (
        t["data_offsets"][0]
            .as_u64()
            .ok_or_else(|| bad("offsets"))? as usize,
        t["data_offsets"][1]
            .as_u64()
            .ok_or_else(|| bad("offsets"))? as usize,
    );
    let data = bytes
        .get(8 + n + a..8 + n + b)
        .ok_or_else(|| bad("data out of range"))?;
    let table = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((table, dim))
}

/// `sha256:` of the canonical json fingerprint, field for field as
/// `PotionEmbedder.fingerprint` writes it (keys sorted, no spaces).
fn model_hash(dir: &Path, dim: usize) -> Result<String, Error> {
    let fp = serde_json::json!({
        "add_special_tokens": false,
        "dtype": "float32-stable",
        "embedder": MODEL_ID,
        "embedding_dim": dim,
        "files_hash": files_hash(dir)?,
        "normalize": "l2",
        "pooling": "mean",
        "tokenizer": "bert-wordpiece-lower",
        "tokenizer_hash": tokenizer_hash(dir)?,
        "version": POTION_VERSION,
    });
    Ok(format!(
        "sha256:{}",
        hex::encode(Sha256::digest(fp.to_string().as_bytes()))
    ))
}

const RELEVANT_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

fn sha256_file(path: &Path) -> Result<String, Error> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

/// sha256 over `name \0 sha256(file) \0` for each relevant file present.
fn files_hash(dir: &Path) -> Result<String, Error> {
    let mut h = Sha256::new();
    for rel in RELEVANT_FILES {
        let p = dir.join(rel);
        if !p.is_file() {
            continue;
        }
        h.update(rel.as_bytes());
        h.update(b"\0");
        h.update(sha256_file(&p)?.as_bytes());
        h.update(b"\0");
    }
    Ok(format!("sha256:{}", hex::encode(h.finalize())))
}

fn tokenizer_hash(dir: &Path) -> Result<String, Error> {
    Ok(format!(
        "sha256:{}",
        sha256_file(&dir.join("tokenizer.json"))?
    ))
}
