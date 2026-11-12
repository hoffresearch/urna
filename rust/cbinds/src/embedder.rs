//! The potion embedder on its own, without a corpus: what the Python forge
//! calls through `ctypes` (urna/_cabi.py) to tokenize and embed with
//! `urna-neural` instead of `tokenizers` and `numpy` (#643).

use std::ffi::CString;
use std::ffi::c_char;
use std::path::Path;

use urna_neural::Potion;

use crate::{arg_str, guarded};

const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Embedder>();
};

/// One loaded potion table.
#[derive(Debug)]
pub struct Embedder {
    potion: Potion,
    model_hash: CString,
}

/// Load the potion table of `model_dir` (`tokenizer.json`,
/// `model.safetensors`, `config.json`). Null on failure, with the reason in
/// `urna_last_error`; close with `urna_embedder_close`.
///
/// # Safety
/// `model_dir` is null or a NUL-terminated string that outlives the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_open(model_dir: *const c_char) -> *mut Embedder {
    guarded(|| {
        // SAFETY: the caller's contract, stated above.
        let dir = unsafe { arg_str(model_dir, "model_dir") }?;
        let potion = Potion::open(Path::new(dir)).map_err(|e| format!("{dir}: {e}"))?;
        let model_hash = CString::new(potion.model_hash()).map_err(|e| e.to_string())?;
        Ok(Embedder { potion, model_hash })
    })
    .map_or(std::ptr::null_mut(), |e| Box::into_raw(Box::new(e)))
}

/// Close an embedder `urna_embedder_open` returned. Null is accepted.
///
/// # Safety
/// `e` is null or a pointer `urna_embedder_open` returned and not yet closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_close(e: *mut Embedder) {
    if e.is_null() {
        return;
    }
    // SAFETY: the pointer came from `Box::into_raw` in `urna_embedder_open`
    // and the caller promises it is closed once.
    drop(unsafe { Box::from_raw(e) });
}

/// The vector width; 0 for null.
///
/// # Safety
/// `e` is null or a live embedder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_dim(e: *const Embedder) -> usize {
    // SAFETY: a null is refused by `as_ref`; else the caller's live embedder.
    unsafe { e.as_ref() }.map_or(0, |e| e.potion.dim())
}

/// The table's `sha256:` model_hash, valid while the embedder is open;
/// "" for null.
///
/// # Safety
/// `e` is null or a live embedder.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_model_hash(e: *const Embedder) -> *const c_char {
    // SAFETY: as above.
    unsafe { e.as_ref() }.map_or(c"".as_ptr(), |e| e.model_hash.as_ptr())
}

/// The WordPiece ids of `text`, no special tokens: up to `cap` of them are
/// written to `ids` and the full count is returned, so a count above `cap`
/// asks for a larger buffer. `usize::MAX` (SIZE_MAX) on failure.
///
/// # Safety
/// `e` is null or a live embedder, `text` a NUL-terminated string, and
/// `ids` null (with `cap` 0) or valid for `cap` writes of `u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_tokenize(
    e: *const Embedder,
    text: *const c_char,
    ids: *mut u32,
    cap: usize,
) -> usize {
    guarded(|| {
        // SAFETY: as above.
        let e = unsafe { e.as_ref() }.ok_or("embedder: null pointer")?;
        // SAFETY: the caller's contract for `text`.
        let text = unsafe { arg_str(text, "text") }?;
        let got = e.potion.token_ids(text);
        if cap > 0 {
            if ids.is_null() {
                return Err("ids: null pointer".to_string());
            }
            // SAFETY: the caller promises `ids` is valid for `cap` writes.
            let out = unsafe { std::slice::from_raw_parts_mut(ids, cap) };
            let n = got.len().min(cap);
            out[..n].copy_from_slice(&got[..n]);
        }
        Ok(got.len())
    })
    .unwrap_or(usize::MAX)
}

/// The unit-norm vector of `text` (the token rows' mean; an empty text is
/// the unk row) into `out`, which holds `urna_embedder_dim` floats. 0 on
/// success, -1 on failure.
///
/// # Safety
/// `e` is null or a live embedder, `text` a NUL-terminated string, and
/// `out` valid for `urna_embedder_dim(e)` writes of `f32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_embedder_embed(
    e: *const Embedder,
    text: *const c_char,
    out: *mut f32,
) -> i32 {
    guarded(|| {
        // SAFETY: as above.
        let e = unsafe { e.as_ref() }.ok_or("embedder: null pointer")?;
        // SAFETY: the caller's contract for `text`.
        let text = unsafe { arg_str(text, "text") }?;
        if out.is_null() {
            return Err("out: null pointer".to_string());
        }
        let v = e.potion.embed(text).map_err(|err| err.to_string())?;
        // SAFETY: the caller promises `out` holds `dim` floats, the length
        // of `v`.
        let dst = unsafe { std::slice::from_raw_parts_mut(out, v.len()) };
        dst.copy_from_slice(&v);
        Ok(0)
    })
    .unwrap_or(-1)
}
