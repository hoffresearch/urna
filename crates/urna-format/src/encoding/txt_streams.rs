//! per-chunk independent canonical-text streams (the `txt_streams` wire
//! codec, encoding id 10). re-layouts the `chunks_canonical` (0x02)
//! section's COMPRESSED form from one concatenated zstd-19 blob into N
//! independently zstd-encoded streams (one per canonical string) plus an
//! intpack offset table that gives O(1) seek to any single chunk.
//!
//! this is the named prerequisite for #11's dict/fsst text levers: a
//! per-chunk frame is where a trained dict or fsst can beat one big
//! zstd-19 blob, and it is the O(1) single-chunk reopen layout (decode ONE
//! stream for cite/materialize instead of inflating the whole section).
//! a per-chunk frame loses cross-chunk LZ context, so expect a small zstd
//! size INCREASE today; the win is what it unlocks. draws from facebook
//! zstd (self-describing per-record frames), lancedb/lance (transparent
//! per-page string compression with a parallel offset table) and
//! flatbuffers (an offset vector giving O(1) random access off mmap).
//!
//! [`decode`] rebuilds the EXACT canonical payload byte-for-byte, so
//! `content_hash` (hashed over decoded bytes) and `urna://` citations are
//! unchanged. every read is bounds-checked and returns a typed
//! `UrnaError`, never a panic on a truncated or hostile payload.
//!
//! container layout (mirrors the intpack/spans-repack discipline; shared by
//! the V1 plain-zstd, V2 dict-framed, and V3 fsst-framed variants):
//!
//! ```text
//! [0]        u8  kind/version  (V1 plain / V2 dict / V3 fsst)
//! [1..9]     u64 chunk count   (LE)
//! [9..9+T]   intpack offset table of N+1 byte offsets into the streams
//!            region (pack_u64s, encoding-id-4 primitive, reused)
//! [9+T..]    N per-stream frames, stream i = bytes [off[i] .. off[i+1])
//! ```

use super::intpack::{IntpackReader, pack_u64s};
use super::zstd_codec::{zstd_decode, zstd_encode};
use crate::bytes::le_u64;
use crate::error::UrnaError;
use crate::layout::{
    SECTION_CHUNKS_CANONICAL, SECTION_PAYLOAD_PREFIX_SIZE, SECTION_PAYLOAD_VERSION,
};

/// leading kind/version byte. v1 = plain per-stream zstd. the dict (V2) and
/// fsst (V3) variants live in `zstd_dict.rs` / `fsst.rs` and claim the next
/// values here, reusing this container without a new encoding id beyond
/// their section-entry codec id.
pub const TXT_STREAMS_V1: u8 = 0;

/// container header: u8 kind + u64 count. shared by all variants.
pub(super) const HEADER: usize = 9;

pub(super) fn malformed(reason: impl Into<String>) -> UrnaError {
    UrnaError::MalformedSectionPayload {
        section_id: SECTION_CHUNKS_CANONICAL,
        reason: reason.into(),
    }
}

/// assemble the shared container: kind byte + count + intpack offset table +
/// the concatenated per-stream frames. a pure function of the inputs, so two
/// builds are byte-identical. shared by the V1/V2/V3 encoders.
pub(super) fn write_container(kind: u8, count: usize, table: &[u8], streams: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + table.len() + streams.len());
    out.push(kind);
    out.extend_from_slice(&(count as u64).to_le_bytes());
    out.extend_from_slice(table);
    out.extend_from_slice(streams);
    out
}

/// rebuild the canonical (raw-encoding) `chunks_canonical` payload from the
/// already-decompressed per-chunk bodies (utf-8 validated by the caller).
/// byte-identical to `sections::encode_chunks_canonical`. shared by all
/// variants so `content_hash` rebuilds the same way regardless of codec.
pub(super) fn build_canonical(count: usize, bodies: &[Vec<u8>]) -> crate::Result<Vec<u8>> {
    let total: usize = bodies.iter().map(|b| b.len()).sum();
    let mut out = Vec::with_capacity(SECTION_PAYLOAD_PREFIX_SIZE + count * 4 + total);
    out.extend_from_slice(&SECTION_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&(count as u64).to_le_bytes());
    for raw in bodies {
        let len = u32::try_from(raw.len())
            .map_err(|_| malformed("txt_streams: stream longer than u32"))?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(raw);
    }
    Ok(out)
}

/// encode `texts` (the canonical strings, in chunk order) as per-chunk
/// independent zstd streams behind an intpack offset table. the layout is
/// a pure function of the inputs, so two builds are byte-identical.
pub fn encode_txt_streams(texts: &[String]) -> crate::Result<Vec<u8>> {
    let mut streams: Vec<u8> = Vec::new();
    // n+1 offsets so stream i is [off[i] .. off[i+1]); off[0] == 0 and the
    // last is the total streams length, giving O(1) seek and exact bounds.
    let mut offsets: Vec<u64> = Vec::with_capacity(texts.len() + 1);
    offsets.push(0);
    for t in texts {
        streams.extend_from_slice(&zstd_encode(t.as_bytes())?);
        offsets.push(streams.len() as u64);
    }
    let table = pack_u64s(&offsets);
    Ok(write_container(
        TXT_STREAMS_V1,
        texts.len(),
        &table,
        &streams,
    ))
}

/// a parsed `txt_streams` payload. `parse` validates the header and offset
/// table once; `stream`/`text` then reach any single chunk in O(1) without
/// touching the others (the O(1)-reopen enabler). `decode_payload` rebuilds
/// the full canonical section by concatenating all of them.
pub struct TxtStreams<'a> {
    streams: &'a [u8],
    offsets: Vec<u64>,
    count: usize,
}

impl<'a> TxtStreams<'a> {
    pub fn parse(bytes: &'a [u8]) -> crate::Result<Self> {
        let (kind, rest) = bytes
            .split_first()
            .ok_or_else(|| malformed("txt_streams: empty"))?;
        if *kind != TXT_STREAMS_V1 {
            return Err(malformed(format!("txt_streams: unknown kind {}", *kind)));
        }
        if rest.len() < 8 {
            return Err(malformed("txt_streams: truncated count"));
        }
        let declared = le_u64(&rest[0..8])?;
        let table_bytes = &rest[8..];
        // the offset table is itself an intpack payload; IntpackReader
        // parses its header/directory and refuses an oversized claim, so the
        // n+1 offsets are the bounded source of truth (never the raw count).
        let reader = IntpackReader::parse(table_bytes)?;
        if reader.is_empty() {
            return Err(malformed("txt_streams: offset table must hold n+1 >= 1"));
        }
        let count = reader.len() - 1;
        // n streams require exactly n+1 offsets; the declared count (checked
        // against the table) must agree without overflowing on a hostile u64.
        if declared != count as u64 {
            return Err(malformed("txt_streams: declared count != offset count - 1"));
        }
        let mut offsets = Vec::with_capacity(reader.len());
        for i in 0..reader.len() {
            offsets.push(reader.get(i)?);
        }
        if offsets[0] != 0 {
            return Err(malformed("txt_streams: first offset must be 0"));
        }
        // the offset table sits before the streams region. pack_u64s output
        // length is a deterministic function of the offsets, so re-pack the
        // parsed offsets and measure to locate the streams start without
        // storing a separate table length.
        let table_len = pack_u64s(&offsets).len();
        if table_len > table_bytes.len() {
            return Err(malformed("txt_streams: truncated offset table"));
        }
        let streams = &table_bytes[table_len..];
        // offsets must be monotonic non-decreasing and end at the streams
        // length, so every stream slice is in-bounds and the layout is exact.
        for w in offsets.windows(2) {
            if w[1] < w[0] {
                return Err(malformed("txt_streams: non-monotonic offsets"));
            }
        }
        if offsets.last().copied() != Some(streams.len() as u64) {
            return Err(malformed("txt_streams: final offset != streams length"));
        }
        Ok(Self {
            streams,
            offsets,
            count,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// the raw zstd bytes of stream `i`, bounds-checked. O(1) seek.
    fn stream(&self, i: usize) -> crate::Result<&'a [u8]> {
        if i >= self.count {
            return Err(malformed("txt_streams: index out of range"));
        }
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        self.streams
            .get(start..end)
            .ok_or_else(|| malformed("txt_streams: stream slice out of bounds"))
    }

    /// decode a SINGLE chunk's canonical text in O(1) (the runtime reopen
    /// path decodes one stream for cite/materialize instead of the whole
    /// section). validates utf-8 so a hostile stream cannot smuggle non-utf8.
    pub fn text(&self, i: usize) -> crate::Result<String> {
        let raw = zstd_decode(self.stream(i)?).map_err(|e| match e {
            UrnaError::MalformedSectionPayload { reason, .. } => malformed(reason),
            other => other,
        })?;
        String::from_utf8(raw).map_err(|e| malformed(format!("txt_streams: invalid utf-8: {}", e)))
    }
}

/// reconstruct the canonical (raw-encoding) `chunks_canonical` payload from
/// a full `txt_streams` payload (including the leading kind byte). the
/// output is byte-identical to `sections::encode_chunks_canonical`, so
/// `content_hash` is preserved. dispatched by `encoding::decode_payload`.
pub fn decode(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    let parsed = TxtStreams::parse(bytes)?;
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(parsed.count);
    for i in 0..parsed.count {
        let raw = zstd_decode(parsed.stream(i)?).map_err(|e| match e {
            UrnaError::MalformedSectionPayload { reason, .. } => malformed(reason),
            other => other,
        })?;
        // validate utf-8 (the raw canonical encoder only ever wrote utf-8;
        // a tampered stream must be rejected, not silently passed through).
        std::str::from_utf8(&raw)
            .map_err(|e| malformed(format!("txt_streams: invalid utf-8: {}", e)))?;
        bodies.push(raw);
    }
    build_canonical(parsed.count, &bodies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sections::encode_chunks_canonical;

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn assert_byte_identical(items: &[&str]) {
        let t = texts(items);
        let packed = encode_txt_streams(&t).unwrap();
        assert_eq!(packed[0], TXT_STREAMS_V1);
        let raw = encode_chunks_canonical(&t).unwrap();
        assert_eq!(decode(&packed).unwrap(), raw, "decode must rebuild raw");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn byte_identical_across_corpora() {
        assert_byte_identical(&[]);
        assert_byte_identical(&["only one"]);
        assert_byte_identical(&["primeiro", "segundo", "terceiro"]);
        // multibyte utf-8 (pt-br accents) must round-trip exactly.
        assert_byte_identical(&["coração", "informação", "açaí é ótimo", ""]);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn o1_seek_returns_the_right_stream() {
        let t = texts(&["alpha", "beta", "gama", "delta"]);
        let packed = encode_txt_streams(&t).unwrap();
        let parsed = TxtStreams::parse(&packed).unwrap();
        assert_eq!(parsed.len(), 4);
        for (i, s) in t.iter().enumerate() {
            assert_eq!(&parsed.text(i).unwrap(), s, "text({}) mismatch", i);
        }
        assert!(parsed.text(4).is_err(), "oob index must error");
    }

    #[test]
    #[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
    fn determinism_two_encodes_byte_identical() {
        let t = texts(&["a", "bb", "ccc", "coração"]);
        assert_eq!(
            encode_txt_streams(&t).unwrap(),
            encode_txt_streams(&t).unwrap()
        );
    }
}
