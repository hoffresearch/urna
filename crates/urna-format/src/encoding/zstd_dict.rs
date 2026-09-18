//! trained-dictionary zstd over per-chunk canonical-text streams (the
//! `zstd_dict` wire codec, encoding id 5). re-frames each `chunks_canonical`
//! (0x02) stream against ONE shared dictionary so a tiny, structurally
//! similar chunk compresses against cross-record statistics instead of a
//! cold per-frame entropy table. the dictionary lives in the optional
//! `SECTION_DICTIONARY` (0x0A), excluded from `content_hash`, so adding it
//! never moves a `urna://` citation.
//!
//! this is the `TXT_STREAMS_V2` variant: it reuses the txt_streams container
//! byte-for-byte (kind byte + count + intpack offset table + N streams,
//! giving O(1) single-chunk reopen) but each stream is zstd-encoded WITH the
//! shared dict via `EncoderDictionary`/`DecoderDictionary`. [`decode`]
//! rebuilds the EXACT canonical payload byte-for-byte, so `content_hash` is
//! preserved. every read is bounds-checked and returns a typed `UrnaError`,
//! never a panic on a truncated or hostile payload.
//!
//! determinism: zstd's ZDICT trainer (`from_continuous`) is a pure function
//! of its sorted sample bytes + sizes, pinned to a fixed zstd version, so
//! two builds over the same corpus produce a byte-identical dictionary (and
//! file). draws from facebook/zstd (ZDICT/COVER trained dictionaries),
//! rocksdb (one cross-block dictionary sampled across many small records),
//! and duckdb (per-section analyze-and-pick over candidate codecs).

use super::intpack::{IntpackReader, pack_u64s};
use super::txt_streams::{build_canonical, malformed, write_container};
use crate::bytes::le_u64;
use crate::error::UrnaError;
use zstd::bulk::{Compressor, Decompressor};
use zstd::dict::{DecoderDictionary, EncoderDictionary};

/// kind/version byte for the dict-framed variant. distinct from
/// `TXT_STREAMS_V1` so a reader dispatches the right per-stream codec
/// without a new encoding id beyond zstd_dict (5) on the section entry.
pub const TXT_STREAMS_V2: u8 = 1;

/// zstd level used for both dict training and per-stream compression. matches
/// `DEFAULT_ZSTD_LEVEL` so the dict path competes on equal footing.
const DICT_LEVEL: i32 = super::zstd_codec::DEFAULT_ZSTD_LEVEL;

/// max trained-dictionary size in bytes (~110 KiB). the ZDICT trainer
/// returns a SMALLER dict when the sample set does not justify the cap, so
/// this is an upper bound, not a fixed size. chosen in the 64-112 KiB band
/// the work order pins; large enough to capture pt-br boilerplate, small
/// enough to stay a negligible slice of the file.
pub const MAX_DICT_BYTES: usize = 112 * 1024;

/// per-stream decompress capacity ceiling when the frame carries no size
/// hint. one canonical chunk is small; this bound just prevents a hostile
/// frame from forcing a giant allocation.
const STREAM_CAP: usize = 64 * 1024 * 1024;

/// train ONE deterministic zstd dictionary over the canonical texts.
///
/// the samples are sorted and deduplicated by the caller (the writer) so the
/// trainer input is canonical; `from_continuous` is then a pure function of
/// the concatenated bytes + per-sample sizes, pinned to the workspace zstd
/// version. returns `None` when there is too little material to train a
/// useful dictionary (the trainer needs a handful of samples); the caller
/// then simply does not offer the dict variant for that build.
pub fn train_dict(sorted_unique: &[String]) -> Option<Vec<u8>> {
    // the trainer rejects empty / too-small corpora; guard so a tiny build
    // (e.g. the golden fixture path) never offers a dict variant.
    if sorted_unique.len() < 8 {
        return None;
    }
    let total: usize = sorted_unique.iter().map(|s| s.len()).sum();
    if total == 0 {
        return None;
    }
    let mut data: Vec<u8> = Vec::with_capacity(total);
    let mut sizes: Vec<usize> = Vec::with_capacity(sorted_unique.len());
    for s in sorted_unique {
        data.extend_from_slice(s.as_bytes());
        sizes.push(s.len());
    }
    // cap the dict proportionately: a dict bigger than ~1/10 of the corpus is
    // pure overhead it can never repay (the chooser counts the dict blob, so
    // an oversized dict just loses). small builds thus train a small dict; the
    // 64-112 KiB band is the ceiling for large corpora. minimum 4 KiB so the
    // trainer has room to capture real boilerplate.
    let cap = (total / 10).clamp(4 * 1024, MAX_DICT_BYTES);
    // zero-length samples are valid input but contribute nothing; the trainer
    // tolerates them. a failed train (corpus too uniform / too small) is not
    // an error: the build just keeps the non-dict candidates.
    zstd::dict::from_continuous(&data, &sizes, cap).ok()
}

/// encode `texts` as per-chunk dict-framed zstd streams behind the shared
/// txt_streams offset table. byte-identical output for identical inputs +
/// identical `dict` (the dict itself is deterministic), so two builds match.
pub fn encode(texts: &[String], dict: &[u8]) -> crate::Result<Vec<u8>> {
    let edict = EncoderDictionary::copy(dict, DICT_LEVEL);
    let mut comp = Compressor::with_prepared_dictionary(&edict)
        .map_err(|e| UrnaError::InvalidInput(format!("zstd_dict: compressor init: {}", e)))?;
    let mut streams: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(texts.len() + 1);
    offsets.push(0);
    for t in texts {
        let c = comp
            .compress(t.as_bytes())
            .map_err(|e| UrnaError::InvalidInput(format!("zstd_dict: compress: {}", e)))?;
        streams.extend_from_slice(&c);
        offsets.push(streams.len() as u64);
    }
    let table = pack_u64s(&offsets);
    Ok(write_container(
        TXT_STREAMS_V2,
        texts.len(),
        &table,
        &streams,
    ))
}

/// decompress a single dict-framed stream with the shared `dict`, validating
/// utf-8 (the canonical encoder only ever wrote utf-8). bounds + typed
/// errors only, never a panic on a hostile frame.
fn decompress_stream(dec: &mut Decompressor<'_>, stream: &[u8]) -> crate::Result<Vec<u8>> {
    let cap = Decompressor::upper_bound(stream)
        .unwrap_or(STREAM_CAP)
        .min(STREAM_CAP);
    let raw = dec
        .decompress(stream, cap)
        .map_err(|e| malformed(format!("zstd_dict: decompress: {}", e)))?;
    std::str::from_utf8(&raw).map_err(|e| malformed(format!("zstd_dict: invalid utf-8: {}", e)))?;
    Ok(raw)
}

/// reconstruct the canonical `chunks_canonical` payload from a dict-framed
/// `txt_streams` V2 payload using the shared `dict`. byte-identical to
/// `sections::encode_chunks_canonical`, so `content_hash` is preserved.
pub fn decode(bytes: &[u8], dict: &[u8]) -> crate::Result<Vec<u8>> {
    let (count, offsets, streams) = parse_v2(bytes)?;
    let ddict = DecoderDictionary::copy(dict);
    let mut dec = Decompressor::with_prepared_dictionary(&ddict)
        .map_err(|e| UrnaError::InvalidInput(format!("zstd_dict: decompressor init: {}", e)))?;
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(count);
    for i in 0..count {
        let s = stream_slice(streams, &offsets, i)?;
        bodies.push(decompress_stream(&mut dec, s)?);
    }
    build_canonical(count, &bodies)
}

/// parse the V2 container header + intpack offset table, returning the
/// chunk count, the n+1 byte offsets, and the streams region. mirrors
/// `TxtStreams::parse` but for the V2 kind byte.
fn parse_v2(bytes: &[u8]) -> crate::Result<(usize, Vec<u64>, &[u8])> {
    let (kind, rest) = bytes
        .split_first()
        .ok_or_else(|| malformed("zstd_dict: empty"))?;
    if *kind != TXT_STREAMS_V2 {
        return Err(malformed(format!("zstd_dict: unknown kind {}", *kind)));
    }
    if rest.len() < 8 {
        return Err(malformed("zstd_dict: truncated count"));
    }
    let declared = le_u64(&rest[0..8])?;
    let table_bytes = &rest[8..];
    let reader = IntpackReader::parse(table_bytes)?;
    if reader.is_empty() {
        return Err(malformed("zstd_dict: offset table must hold n+1 >= 1"));
    }
    let count = reader.len() - 1;
    if declared != count as u64 {
        return Err(malformed("zstd_dict: declared count != offset count - 1"));
    }
    let mut offsets = Vec::with_capacity(reader.len());
    for i in 0..reader.len() {
        offsets.push(reader.get(i)?);
    }
    if offsets[0] != 0 {
        return Err(malformed("zstd_dict: first offset must be 0"));
    }
    let table_len = pack_u64s(&offsets).len();
    if table_len > table_bytes.len() {
        return Err(malformed("zstd_dict: truncated offset table"));
    }
    let streams = &table_bytes[table_len..];
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return Err(malformed("zstd_dict: non-monotonic offsets"));
        }
    }
    if offsets.last().copied() != Some(streams.len() as u64) {
        return Err(malformed("zstd_dict: final offset != streams length"));
    }
    Ok((count, offsets, streams))
}

#[inline]
fn stream_slice<'a>(streams: &'a [u8], offsets: &[u64], i: usize) -> crate::Result<&'a [u8]> {
    let start = offsets[i] as usize;
    let end = offsets[i + 1] as usize;
    streams
        .get(start..end)
        .ok_or_else(|| malformed("zstd_dict: stream slice out of bounds"))
}
