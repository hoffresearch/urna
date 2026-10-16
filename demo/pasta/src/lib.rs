//! Pasta Cards core: a `.urna` file and its card fields in, card search out.
//! Everything at query time runs in process: type-ahead over names, hybrid
//! text search with the potion table, deck filters, the card's AVIF bytes.

pub mod card;
pub mod complete;
pub mod filter;
pub mod potion;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use urna_engine::MmapUrnaFile;

pub use card::{Card, Kind};
pub use complete::Suggestion;
pub use filter::Filter;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("urna: {0}")]
    Urna(#[from] urna_engine::RuntimeError),
    #[error("model: {0}")]
    Model(String),
    #[error("fields: {0}")]
    Fields(String),
    #[error("media: {0}")]
    Media(String),
    #[error("the file was embedded with {file}, the potion table here is {ours}")]
    ModelMismatch { file: String, ours: String },
    #[error("the fields file belongs to {fields}, the .urna is {urna}")]
    ContentMismatch { fields: String, urna: String },
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub ordinal: usize,
    pub score: f32,
    /// `name` when the query is a card name, `hybrid` for a text search,
    /// `browse` for filters with no text, ranked by popularity.
    pub via: &'static str,
}

pub struct Timings {
    pub open_ms: f64,
    pub fields_ms: f64,
    pub index_ms: f64,
    pub potion_ms: f64,
}

pub struct Pasta {
    file: MmapUrnaFile,
    cards: Vec<Card>,
    by_chunk_id: HashMap<String, usize>,
    by_name: HashMap<String, Vec<usize>>,
    blob_of: Vec<Option<usize>>,
    completer: complete::Completer,
    potion: potion::Potion,
    pub timings: Timings,
}

impl Pasta {
    pub fn open(urna: &Path, fields: &Path, potion_dir: &Path) -> Result<Self, Error> {
        let t = Instant::now();
        let file = MmapUrnaFile::open(urna)?;
        let open_ms = ms(t);

        let t = Instant::now();
        let cards = read_fields(fields, file.content_hash(), file.chunk_ids().len())?;
        let fields_ms = ms(t);

        let t = Instant::now();
        let by_chunk_id = file.chunk_ids().iter().enumerate().map(|(i, id)| (id.clone(), i)).collect();
        let mut by_name: HashMap<String, Vec<usize>> = HashMap::new();
        for c in &cards {
            by_name.entry(complete::fold(&c.name)).or_default().push(c.ordinal);
            if let Some(pt) = &c.pt_name {
                by_name.entry(complete::fold(pt)).or_default().push(c.ordinal);
            }
        }
        let blob_of = blob_index(&file, cards.len());
        let completer = complete::Completer::new(&cards);
        let index_ms = ms(t);

        let t = Instant::now();
        let potion = potion::Potion::open(potion_dir)?;
        if potion.model_hash() != file.model_hash() {
            return Err(Error::ModelMismatch { file: file.model_hash().into(), ours: potion.model_hash().into() });
        }
        let potion_ms = ms(t);

        Ok(Self {
            file,
            cards,
            by_chunk_id,
            by_name,
            blob_of,
            completer,
            potion,
            timings: Timings { open_ms, fields_ms, index_ms, potion_ms },
        })
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards
    }

    pub fn card(&self, ordinal: usize) -> Option<&Card> {
        self.cards.get(ordinal)
    }

    pub fn complete(&self, query: &str, k: usize) -> Vec<Suggestion> {
        self.completer.complete(&self.cards, query, k)
    }

    /// A card name in English or Portuguese answers with that card first;
    /// other text goes through the hybrid path (potion + BM25, exact cosine
    /// rerank); no text with filters browses by popularity.
    pub fn search(&self, query: &str, filter: &Filter, k: usize) -> Result<Vec<Hit>, Error> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(self.browse(filter, k));
        }
        let mut hits: Vec<Hit> = Vec::with_capacity(k);
        if let Some(named) = self.by_name.get(&complete::fold(query)) {
            let named = named.iter().copied().filter(|o| filter.keeps(&self.cards[*o]));
            hits.extend(named.map(|ordinal| Hit { ordinal, score: 1.0, via: "name" }));
        }
        let vec = self.potion.embed(query)?;
        // filters drop rows after the rerank, so a narrow filter fetches more.
        let over = if filter.is_default() { (k * 4).max(40) } else { (k * 40).max(400) };
        let res = self.file.search_hybrid(&vec, query, over as i32, over.min(256))?;
        for h in res.hits {
            let Some(&ordinal) = self.by_chunk_id.get(&h.chunk_id) else { continue };
            if filter.keeps(&self.cards[ordinal]) && !hits.iter().any(|x| x.ordinal == ordinal) {
                hits.push(Hit { ordinal, score: h.score, via: "hybrid" });
            }
        }
        hits.truncate(k);
        Ok(hits)
    }

    fn browse(&self, filter: &Filter, k: usize) -> Vec<Hit> {
        let mut kept: Vec<&Card> = self.cards.iter().filter(|c| filter.keeps(c)).collect();
        kept.sort_by_key(|c| (c.edhrec_rank.unwrap_or(u32::MAX), c.ordinal));
        kept.into_iter().take(k).map(|c| Hit { ordinal: c.ordinal, score: 0.0, via: "browse" }).collect()
    }

    /// The card's image as stored: AVIF bytes, sliced off the mmap with no
    /// copy. `None` when the file keeps its media as one stream. iOS,
    /// Android and the webviews decode these bytes natively.
    pub fn image(&self, ordinal: usize) -> Option<&[u8]> {
        let blob = (*self.blob_of.get(ordinal)?)?;
        self.file.blob_bytes(blob).ok()
    }

    /// The card's image decoded to pixels in process (dav1d), for the
    /// desktop and the terminal.
    #[cfg(feature = "decode")]
    pub fn decode_image(&self, ordinal: usize) -> Result<image::DynamicImage, Error> {
        let bytes = self.image(ordinal).ok_or_else(|| Error::Media(format!("no image for card {ordinal}")))?;
        image::load_from_memory_with_format(bytes, image::ImageFormat::Avif)
            .map_err(|e| Error::Media(format!("card {ordinal}: {e}")))
    }
}

fn read_fields(path: &Path, content_hash: &str, n: usize) -> Result<Vec<Card>, Error> {
    let raw = std::fs::read(path).map_err(|e| Error::Fields(format!("{}: {e}", path.display())))?;
    let json = zstd::decode_all(raw.as_slice()).map_err(|e| Error::Fields(format!("zstd: {e}")))?;
    let doc: card::FieldsFile = serde_json::from_slice(&json).map_err(|e| Error::Fields(format!("json: {e}")))?;
    if doc.format != "pasta-fields/1" {
        return Err(Error::Fields(format!("unknown format {}", doc.format)));
    }
    if doc.content_hash != content_hash {
        return Err(Error::ContentMismatch { fields: doc.content_hash, urna: content_hash.into() });
    }
    if doc.cards.len() != n || doc.cards.iter().enumerate().any(|(i, c)| c.ordinal != i) {
        return Err(Error::Fields(format!("expected ordinals 0..{n} in order")));
    }
    Ok(doc.cards)
}

/// Per-card blobs are named `<dataset>/<ordinal>.avif`; a file with one
/// media stream maps nothing.
fn blob_index(file: &MmapUrnaFile, n: usize) -> Vec<Option<usize>> {
    let mut out = vec![None; n];
    for (i, r) in file.blob_refs().unwrap_or_default().iter().enumerate() {
        let stem = r.original_uri.rsplit('/').next().and_then(|f| f.strip_suffix(".avif"));
        if let Some(o) = stem.and_then(|s| s.parse::<usize>().ok())
            && o < n
        {
            out[o] = Some(i);
        }
    }
    out
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
