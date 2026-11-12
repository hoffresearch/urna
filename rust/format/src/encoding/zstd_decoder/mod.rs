//! our zstd decoder (RFC 8878), in safe Rust: the read path of a `.urna`
//! links no C. it decodes what the `zstd` crate's encoder writes (levels,
//! long windows, trained or raw-content dictionaries, checksums, frames
//! back to back) and refuses everything else with an error, never a panic.
//!
//! every decode runs under a hard output limit: a frame that declares more
//! is refused before anything is allocated, and one that keeps producing
//! stops at the limit, so a tiny hostile frame cannot inflate into memory.
//! the encoder and the dictionary trainer stay on the `zstd` crate, behind
//! the `write` feature; the parity workspace (`rust/format/parity`) holds
//! this decoder to libzstd byte for byte.
//!
//! the files follow the format: `frame` (frames and blocks), `literals` and
//! `huffman` (the literals section), `sequences` and `codes` (the sequences
//! section), `fse` (the entropy tables both use), `dictionary`, `bits` (the
//! two bit readers) and `xxhash` (the content checksum).

mod bits;
mod codes;
mod dictionary;
mod frame;
mod fse;
mod huffman;
mod literals;
mod sequences;
mod xxhash;

#[cfg(test)]
mod tests;

use dictionary::Dictionary;

/// why a zstd payload was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ZstdError(&'static str);

impl std::fmt::Display for ZstdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zstd: {}", self.0)
    }
}

impl std::error::Error for ZstdError {}

type Res<T> = Result<T, ZstdError>;

fn corrupt(reason: &'static str) -> ZstdError {
    ZstdError(reason)
}

/// make room for `extra` more bytes in `out` without ever letting it hold
/// more than `limit`: the limit is checked first, and the capacity grows
/// geometrically but never past the limit (`try_reserve`, so a failed
/// allocation is an error too, not an abort).
fn make_room(out: &mut Vec<u8>, extra: usize, limit: usize) -> Res<()> {
    let need = out
        .len()
        .checked_add(extra)
        .filter(|&need| need <= limit)
        .ok_or(corrupt("output exceeds the size limit"))?;
    if need <= out.capacity() {
        return Ok(());
    }
    let target = out
        .capacity()
        .saturating_mul(2)
        .clamp(need, limit.max(need));
    out.try_reserve_exact(target - out.len())
        .map_err(|_| corrupt("output allocation failed"))
}

/// a decoder with its scratch buffers, optionally bound to a dictionary;
/// reuse one across many small frames.
pub(crate) struct ZstdDecoder<'d> {
    dict: Option<Dictionary<'d>>,
    scratch: frame::Scratch,
}

impl std::fmt::Debug for ZstdDecoder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZstdDecoder")
            .field("dict", &self.dict)
            .finish_non_exhaustive()
    }
}

impl Default for ZstdDecoder<'_> {
    fn default() -> Self {
        Self {
            dict: None,
            scratch: frame::Scratch::default(),
        }
    }
}

impl<'d> ZstdDecoder<'d> {
    /// a decoder for frames compressed against `dict` (a trained dictionary
    /// or raw content). frames that name another dictionary ID are refused.
    pub(crate) fn with_dictionary(dict: &'d [u8]) -> Result<Self, ZstdError> {
        Ok(Self {
            dict: Some(Dictionary::parse(dict)?),
            scratch: frame::Scratch::default(),
        })
    }

    /// decode every frame of `src`, producing at most `limit` bytes.
    pub(crate) fn decompress(&mut self, src: &[u8], limit: usize) -> Result<Vec<u8>, ZstdError> {
        let mut out = Vec::new();
        frame::decode_all(src, self.dict.as_ref(), limit, &mut out, &mut self.scratch)?;
        Ok(out)
    }
}
