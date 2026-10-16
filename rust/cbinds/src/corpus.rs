//! The corpus handle: the engine's mmap, the stored texts, the chunk to
//! image-blob map and the potion embedder, loaded on the first ask from
//! the model dir given at open (or `URNA_POTION_DIR`).

use std::collections::HashMap;
use std::ffi::c_char;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use urna_engine::MmapUrnaFile;
use urna_engine::scalars::parse;
use urna_neural::Potion;

use crate::hits::{Hit, Hits};
use crate::{arg_str, give_string, guarded, set_error};

/// One opened `.urna`.
#[derive(Debug)]
pub struct Corpus {
    rt: MmapUrnaFile,
    texts: Vec<String>,
    index: HashMap<String, usize>,
    /// chunk -> the inlined image blob its source names, when one.
    blob_of: Vec<Option<usize>>,
    model_dir: Option<PathBuf>,
    potion: OnceLock<Result<Potion, String>>,
}

impl Corpus {
    fn open(path: &Path, model_dir: Option<PathBuf>) -> Result<Self, String> {
        let rt = MmapUrnaFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let texts = rt.canonical_texts().map_err(|e| e.to_string())?;
        let index = rt
            .chunk_ids()
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let blobs: HashMap<&str, usize> = rt
            .blob_refs()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.inlined)
            .map(|(i, r)| (r.original_uri.as_str(), i))
            .collect();
        let blob_of = (0..texts.len())
            .map(|i| {
                rt.span(i)
                    .and_then(|s| blobs.get(s.source_uri.as_str()).copied())
            })
            .collect();
        Ok(Self {
            rt,
            texts,
            index,
            blob_of,
            model_dir,
            potion: OnceLock::new(),
        })
    }

    fn potion(&self) -> Result<&Potion, String> {
        self.potion
            .get_or_init(|| {
                let model = self.rt.embedding_model();
                if !model.starts_with("minishlab/potion") {
                    return Err(format!(
                        "the corpus was embedded with {model}; this ABI embeds potion corpora only"
                    ));
                }
                let dir = self
                    .model_dir
                    .clone()
                    .or_else(|| std::env::var_os("URNA_POTION_DIR").map(PathBuf::from))
                    .ok_or("no model dir: pass it to urna_open or set URNA_POTION_DIR")?;
                let p = Potion::open(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                if p.model_hash() != self.rt.model_hash() {
                    return Err(format!(
                        "model_hash mismatch: corpus {}, table {}",
                        self.rt.model_hash(),
                        p.model_hash()
                    ));
                }
                if p.dim() != self.rt.embedding_dim() {
                    return Err(format!(
                        "dim mismatch: corpus {}, table {}",
                        self.rt.embedding_dim(),
                        p.dim()
                    ));
                }
                Ok(p)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn ask(&self, query: &str, k: usize) -> Result<Hits, String> {
        if k == 0 {
            return Err("k must be at least 1".into());
        }
        let schema = self.rt.scalars().map_or(&[][..], |s| s.schema.as_slice());
        let parsed = parse(query, schema).map_err(|e| e.to_string())?;
        if parsed.text.trim().is_empty() {
            return Err(
                "the query has no text to search; predicates alone are not a search".into(),
            );
        }
        let keep = if parsed.filter.is_empty() {
            None
        } else {
            Some(self.rt.keep(&parsed.filter).map_err(|e| e.to_string())?)
        };
        let vector = self
            .potion()?
            .embed(&parsed.text)
            .map_err(|e| e.to_string())?;
        let k = i32::try_from(k).map_err(|_| "k is too large".to_string())?;
        let cand = (k as usize * 4).max(64);
        let result = self
            .rt
            .search_routed_where(&vector, Some(&parsed.text), k, cand, keep.as_ref())
            .map_err(|e| e.to_string())?;
        let items = result
            .hits
            .iter()
            .filter_map(|h| {
                let chunk = *self.index.get(&h.chunk_id)?;
                Some(Hit::new(
                    chunk,
                    h.score,
                    self.texts.get(chunk).map_or("", String::as_str),
                    &h.citation_id,
                ))
            })
            .collect();
        Ok(Hits { items })
    }
}

/// Open the corpus at `path`. `model_dir` is the potion table dir, or null
/// to take `URNA_POTION_DIR` on the first ask. Null on failure, with the
/// reason in `urna_last_error`.
///
/// # Safety
/// Both pointers are null or NUL-terminated strings that outlive the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_open(path: *const c_char, model_dir: *const c_char) -> *mut Corpus {
    guarded(|| {
        // SAFETY: the caller's contract, stated above.
        let path = unsafe { arg_str(path, "path") }?;
        let dir = if model_dir.is_null() {
            None
        } else {
            // SAFETY: as above, for the second string.
            Some(PathBuf::from(unsafe { arg_str(model_dir, "model_dir") }?))
        };
        Corpus::open(Path::new(path), dir)
    })
    .map_or(std::ptr::null_mut(), |c| Box::into_raw(Box::new(c)))
}

/// Close a corpus `urna_open` returned. Null is accepted. Every `Hits` of
/// it must be freed before, and no pointer `urna_image` returned is read
/// after.
///
/// # Safety
/// `c` is null or a pointer `urna_open` returned and not yet closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_close(c: *mut Corpus) {
    if c.is_null() {
        return;
    }
    // SAFETY: the pointer came from `Box::into_raw` in `urna_open` and the
    // caller promises it is closed once.
    drop(unsafe { Box::from_raw(c) });
}

/// How many chunks the corpus holds; 0 for a null corpus.
///
/// # Safety
/// `c` is null or a live corpus.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_n_chunks(c: *const Corpus) -> usize {
    // SAFETY: a null is refused; otherwise the caller promises a live corpus.
    unsafe { c.as_ref() }.map_or(0, |c| c.texts.len())
}

/// Ask `query` (free text plus the schema's `prefix:value` terms) for the
/// top `k` hits. Null on failure, with the reason in `urna_last_error`;
/// free the answer with `urna_hits_free`.
///
/// # Safety
/// `c` is a live corpus; `query` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_ask(c: *const Corpus, query: *const c_char, k: u32) -> *mut Hits {
    guarded(|| {
        // SAFETY: the caller promises a live corpus or null, refused here.
        let corpus = unsafe { c.as_ref() }.ok_or("corpus: null pointer")?;
        // SAFETY: the caller's contract for the string.
        let query = unsafe { arg_str(query, "query") }?;
        corpus.ask(query, k as usize)
    })
    .map_or(std::ptr::null_mut(), |h| Box::into_raw(Box::new(h)))
}

/// The stored image of chunk `chunk` (an AVIF the platform decodes): its
/// bytes, `*len` long, valid while the corpus is open. Null when the chunk
/// has no inlined image, or on error (then `urna_last_error` says).
///
/// # Safety
/// `c` is a live corpus; `len` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_image(c: *const Corpus, chunk: usize, len: *mut usize) -> *const u8 {
    let out = guarded(|| {
        // SAFETY: the caller promises a live corpus or null, refused here.
        let corpus = unsafe { c.as_ref() }.ok_or("corpus: null pointer")?;
        let Some(blob) = corpus.blob_of.get(chunk).copied().flatten() else {
            return Ok(None);
        };
        corpus
            .rt
            .blob_bytes(blob)
            .map(Some)
            .map_err(|e| e.to_string())
    })
    .flatten();
    if !len.is_null() {
        // SAFETY: the caller promises `len` is writable when not null.
        unsafe { *len = out.map_or(0, <[u8]>::len) };
    }
    out.map_or(std::ptr::null(), <[u8]>::as_ptr)
}

/// The manifest schema as a JSON array (empty array without one), to be
/// freed with `urna_string_free`. Null on a null corpus.
///
/// # Safety
/// `c` is null or a live corpus.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_schema_json(c: *const Corpus) -> *mut c_char {
    guarded(|| {
        // SAFETY: the caller promises a live corpus or null, refused here.
        let corpus = unsafe { c.as_ref() }.ok_or("corpus: null pointer")?;
        let schema = corpus.rt.scalars().map_or(&[][..], |s| s.schema.as_slice());
        serde_json::to_string(schema).map_err(|e| e.to_string())
    })
    .map_or(std::ptr::null_mut(), give_string)
}

/// The corpus's content hash (`sha256:…`), valid while the corpus is open.
///
/// # Safety
/// `c` is null or a live corpus.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_content_hash(c: *const Corpus) -> *mut c_char {
    // SAFETY: the caller promises a live corpus or null, refused here.
    match unsafe { c.as_ref() } {
        Some(corpus) => give_string(corpus.rt.content_hash().to_string()),
        None => {
            set_error("corpus: null pointer");
            std::ptr::null_mut()
        }
    }
}
