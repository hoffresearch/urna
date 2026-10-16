//! An answer: the hits of one ask, each with its chunk index, exact cosine
//! score, stored text and citation, read by index until freed.

use std::ffi::{CString, c_char};

/// One hit, with its strings already NUL-terminated for C.
#[derive(Debug)]
pub(crate) struct Hit {
    chunk: usize,
    score: f32,
    text: CString,
    citation: CString,
}

impl Hit {
    pub(crate) fn new(chunk: usize, score: f32, text: &str, citation: &str) -> Self {
        let clean = |s: &str| CString::new(s.replace('\0', "")).unwrap_or_default();
        Self {
            chunk,
            score,
            text: clean(text),
            citation: clean(citation),
        }
    }
}

/// The hits of one ask.
#[derive(Debug)]
pub struct Hits {
    pub(crate) items: Vec<Hit>,
}

/// How many hits; 0 for null.
///
/// # Safety
/// `h` is null or a live answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hits_len(h: *const Hits) -> usize {
    // SAFETY: null is refused by `as_ref`; else the caller's live answer.
    unsafe { h.as_ref() }.map_or(0, |h| h.items.len())
}

/// The chunk index of hit `i`, or -1 when out of range.
///
/// # Safety
/// `h` is null or a live answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hit_chunk(h: *const Hits, i: usize) -> i64 {
    // SAFETY: as above.
    unsafe { h.as_ref() }
        .and_then(|h| h.items.get(i))
        .map_or(-1, |hit| i64::try_from(hit.chunk).unwrap_or(-1))
}

/// The exact cosine score of hit `i`, or NaN when out of range.
///
/// # Safety
/// `h` is null or a live answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hit_score(h: *const Hits, i: usize) -> f32 {
    // SAFETY: as above.
    unsafe { h.as_ref() }
        .and_then(|h| h.items.get(i))
        .map_or(f32::NAN, |hit| hit.score)
}

/// The stored canonical text of hit `i`, valid until the answer is freed;
/// null when out of range.
///
/// # Safety
/// `h` is null or a live answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hit_text(h: *const Hits, i: usize) -> *const c_char {
    // SAFETY: as above.
    unsafe { h.as_ref() }
        .and_then(|h| h.items.get(i))
        .map_or(std::ptr::null(), |hit| hit.text.as_ptr())
}

/// The `urna://content_hash/chunk_id` citation of hit `i`, valid until the
/// answer is freed; null when out of range.
///
/// # Safety
/// `h` is null or a live answer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hit_citation(h: *const Hits, i: usize) -> *const c_char {
    // SAFETY: as above.
    unsafe { h.as_ref() }
        .and_then(|h| h.items.get(i))
        .map_or(std::ptr::null(), |hit| hit.citation.as_ptr())
}

/// Free an answer `urna_ask` returned. Null is accepted.
///
/// # Safety
/// `h` is null or a pointer `urna_ask` returned and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_hits_free(h: *mut Hits) {
    if h.is_null() {
        return;
    }
    // SAFETY: the pointer came from `Box::into_raw` in `urna_ask` and the
    // caller promises it is freed once.
    drop(unsafe { Box::from_raw(h) });
}
