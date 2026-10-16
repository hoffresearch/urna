//! One corpus opened for the life of the explorer: the mmap the queries run
//! on, the stored text of every chunk, the image blobs by uri, the labels
//! the type-ahead offers and, when the corpus carries them beside it, the
//! card fields the deck filters read. Opening hashes the whole file, so it
//! happens once, on the loader thread, and every query after reuses it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use urna_engine::MmapUrnaFile;

use super::fields::{self, Card};
use super::filter::Filter;
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
    /// one card per chunk ordinal, when the fields file sits beside the corpus.
    pub cards: Option<Vec<Card>>,
    /// folded label and folded alias -> the chunk indices that carry it, so
    /// a query that spells a label finds it even when it ranks low.
    names: HashMap<String, Vec<usize>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("texts", &self.texts.len())
            .field("images", &self.images.len())
            .field("cards", &self.cards.as_ref().map(Vec::len))
            .finish_non_exhaustive()
    }
}

impl Session {
    #[cfg(test)]
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_runtime(MmapUrnaFile::open(path)?, path)
    }

    /// `corpus_path` is where the fields file is looked for; a file beside
    /// the corpus that does not belong to it is an error, not a silent skip.
    pub fn from_runtime(rt: MmapUrnaFile, corpus_path: &Path) -> Result<Self> {
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
        let cards = match &fields::locate(corpus_path) {
            Some(p) => Some(fields::read(p, rt.content_hash(), rt.chunk_ids().len())?),
            None => None,
        };
        let labels = match &cards {
            Some(cards) => Labels::from_cards(cards),
            None => Labels::new(
                rt.chunk_ids()
                    .iter()
                    .map(|id| texts.get(id).map_or("", String::as_str)),
            ),
        };
        let mut names: HashMap<String, Vec<usize>> = HashMap::new();
        let mut name = |key: String, i: usize| {
            let slot = names.entry(key).or_default();
            if !slot.contains(&i) {
                slot.push(i);
            }
        };
        for (i, id) in rt.chunk_ids().iter().enumerate() {
            let first = texts
                .get(id)
                .and_then(|t| t.lines().next())
                .map(str::trim)
                .unwrap_or("");
            if !first.is_empty() {
                let folded = fold(first);
                if let Some(a) = alias(&folded) {
                    name(a.to_string(), i);
                }
                name(folded, i);
            }
            if let Some(c) = cards.as_ref().and_then(|c| c.get(i)) {
                name(fold(&c.name), i);
                if let Some(pt) = &c.pt_name {
                    name(fold(pt), i);
                }
            }
        }
        let chunker = rt.chunker_version().to_string();
        Ok(Self {
            rt,
            texts,
            images,
            labels,
            chunker,
            cards,
            names,
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

    /// The card at chunk ordinal `chunk`, when the corpus carries fields.
    pub fn card(&self, chunk: usize) -> Option<&Card> {
        self.cards.as_ref()?.get(chunk)
    }

    /// Whether `filter` keeps chunk `chunk`; a corpus without fields keeps
    /// everything.
    pub fn keeps(&self, chunk: usize, filter: &Filter) -> bool {
        self.card(chunk).is_none_or(|c| filter.keeps(c))
    }
}
