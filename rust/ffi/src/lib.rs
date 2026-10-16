//! The C ABI the native apps call: open a corpus once, ask it in the search
//! syntax its schema declares, read the hits' stored text and citations,
//! take a hit's image bytes for the platform to decode, read the schema as
//! JSON, close. One opaque handle per corpus and per answer; every string
//! crossing the boundary is UTF-8 with a NUL; a null pointer is refused,
//! never dereferenced; no panic crosses the boundary (`catch_unwind`), it
//! becomes a null return and a message in `urna_last_error`.
//!
//! Thread model: a `Corpus` may be shared by threads for `urna_ask` and
//! `urna_image` (the engine is read-only); `urna_close` must be the last
//! call. `urna_last_error` is per thread.

mod corpus;
mod hits;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

pub use corpus::Corpus;
pub use hits::Hits;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub(crate) fn set_error(msg: impl Into<String>) {
    let text: String = msg.into();
    // a NUL inside the message would truncate it; strip it rather than fail.
    let clean: String = text.chars().filter(|c| *c != '\0').collect();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(clean).unwrap_or_default();
    });
}

/// Run `f`, turning a panic into an error message and `None`.
pub(crate) fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Some(v),
        Ok(Err(e)) => {
            set_error(e);
            None
        }
        Err(_) => {
            set_error("urna: internal panic (reported, not propagated)");
            None
        }
    }
}

/// A C string argument as `&str`, or the reason it is not.
///
/// # Safety
/// `p` is null or points to a NUL-terminated string that outlives the call.
pub(crate) unsafe fn arg_str<'a>(p: *const c_char, what: &str) -> Result<&'a str, String> {
    if p.is_null() {
        return Err(format!("{what}: null pointer"));
    }
    // SAFETY: the caller promises a NUL-terminated string that outlives
    // the call; the null case was refused above.
    let cs = unsafe { CStr::from_ptr(p) };
    cs.to_str().map_err(|_| format!("{what}: not utf-8"))
}

/// The last error of this thread, as a NUL-terminated string valid until
/// the next call that fails on this thread. Empty when nothing failed.
#[unsafe(no_mangle)]
pub extern "C" fn urna_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Free a string `urna_schema_json` returned. Null is accepted.
///
/// # Safety
/// `s` is null or a pointer this library returned and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn urna_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: the pointer came from `CString::into_raw` in this library and
    // the caller promises it is freed once.
    drop(unsafe { CString::from_raw(s) });
}

/// Hand a string to C: the caller frees it with `urna_string_free`.
pub(crate) fn give_string(s: String) -> *mut c_char {
    let clean: String = s.chars().filter(|c| *c != '\0').collect();
    CString::new(clean)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// The version of the library, as `MAJOR.MINOR.PATCH`, static.
#[unsafe(no_mangle)]
pub extern "C" fn urna_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_per_thread_strings_without_nul() {
        set_error("boom\0 trailing");
        // SAFETY: urna_last_error returns a valid NUL-terminated string.
        let s = unsafe { CStr::from_ptr(urna_last_error()) };
        assert_eq!(s.to_str().ok(), Some("boom trailing"));
        let other = std::thread::spawn(|| {
            // SAFETY: as above, on another thread with its own slot.
            unsafe { CStr::from_ptr(urna_last_error()) }
                .to_str()
                .map(str::to_string)
                .ok()
        })
        .join()
        .ok()
        .flatten();
        assert_eq!(other.as_deref(), Some(""));
        assert!(guarded::<()>(|| panic!("x")).is_none());
        // SAFETY: urna_version returns a static NUL-terminated string.
        let v = unsafe { CStr::from_ptr(urna_version()) };
        assert!(v.to_str().is_ok_and(|v| v.contains('.')));
    }
}
