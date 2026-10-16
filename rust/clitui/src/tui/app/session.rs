//! One corpus opened for the life of the explorer: the mmap the queries run
//! on, the stored text of every chunk, the image blobs by uri and the labels
//! the type-ahead offers. Opening hashes the whole file, so it happens once,
//! on the loader thread, and every query after reuses it.

use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use anyhow::Result;
use urna_engine::MmapUrnaFile;

use super::labels::{Labels, alias, fold};

/// The image formats the ask tab can draw, by the blob uri's extension.
const IMAGE_EXT: [&str; 1] = [".avif"];

pub struct Session {
    pub rt: MmapUrnaFile,
    /// chunk_id -> stored canonical text (tier-1, what `cite` returns).
    pub texts: HashMap<String, String>,
    /// blob uri -> blob index, for the inlined blobs that are images.
    pub images: HashMap<String, usize>,
    pub labels: Labels,
    /// the template the stored text follows (`card::knows`).
    pub chunker: String,
    /// folded label and folded alias -> the chunk indices that carry it, so
    /// a query that spells a label finds it even when it ranks low.
    names: HashMap<String, Vec<usize>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("texts", &self.texts.len())
            .field("images", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl Session {
    #[cfg(test)]
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_runtime(MmapUrnaFile::open(path)?)
    }

    pub fn from_runtime(rt: MmapUrnaFile) -> Result<Self> {
        let texts: HashMap<String, String> = rt
            .chunk_ids()
            .iter()
            .cloned()
            .zip(rt.canonical_texts()?)
            .collect();
        let images = rt
            .blob_refs()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.inlined && IMAGE_EXT.iter().any(|e| r.original_uri.ends_with(e)))
            .map(|(i, r)| (r.original_uri.clone(), i))
            .collect();
        let labels = Labels::new(texts.values().map(String::as_str));
        let mut names: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, id) in rt.chunk_ids().iter().enumerate() {
            let first = texts
                .get(id)
                .and_then(|t| t.lines().next())
                .map(str::trim)
                .unwrap_or("");
            if first.is_empty() {
                continue;
            }
            let folded = fold(first);
            if let Some(a) = alias(&folded) {
                names.entry(a.to_string()).or_default().push(i);
            }
            names.entry(folded).or_default().push(i);
        }
        let chunker = rt.chunker_version().to_string();
        Ok(Self {
            rt,
            texts,
            images,
            labels,
            names,
            chunker,
        })
    }

    /// The chunk indices whose label or alias is exactly `query`, in file order.
    pub fn named(&self, query: &str) -> &[usize] {
        self.names
            .get(&fold(query.trim()))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// The image blob a hit's source points at, when it is one.
    pub fn image_of(&self, source_uri: &str) -> Option<usize> {
        self.images.get(source_uri).copied()
    }
}
