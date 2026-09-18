//! `chunks_original_spans` section (`SECTION_CHUNKS_ORIGINAL_SPANS = 0x03`).
//! `(source_uri, byte_start, byte_end)` per chunk — the offset into the
//! original source document the chunk text came from. Required for
//! citation resolution.
//!
//! the raw encoding repeats the full `source_uri` for every chunk. the
//! `intpack` repack (encoding id 4, kind 1) dedups the uris into a pool
//! and bitpacks `(uri_index, byte_start, byte_end - byte_start)`, then
//! reconstructs the exact raw payload on decode, so `content_hash` is
//! byte-identical and this canonical section is never version-bumped.

use std::collections::HashMap;

use super::REPACK_KIND_SPANS;
use super::codec::{Cursor, read_prefix, write_lp_str, write_prefix};
use crate::encoding::{pack_u64s, unpack_u64s};
use crate::error::UrnaError;
use crate::layout::SECTION_CHUNKS_ORIGINAL_SPANS;

#[derive(Clone, Debug, PartialEq)]
pub struct OriginalSpan {
    pub source_uri: String,
    pub byte_start: u64,
    pub byte_end: u64,
}

pub fn encode_chunks_original_spans(spans: &[OriginalSpan]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, spans.len() as u64);
    for s in spans {
        write_lp_str(&mut buf, &s.source_uri)?;
        buf.extend_from_slice(&s.byte_start.to_le_bytes());
        buf.extend_from_slice(&s.byte_end.to_le_bytes());
    }
    Ok(buf)
}

pub fn decode_chunks_original_spans(
    data: &[u8],
    expected_count: usize,
) -> crate::Result<Vec<OriginalSpan>> {
    let mut c = Cursor::new(data, SECTION_CHUNKS_ORIGINAL_SPANS);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(UrnaError::SectionCountMismatch {
            section_id: SECTION_CHUNKS_ORIGINAL_SPANS,
            expected: expected_count,
            got: count,
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let source_uri = c.read_lp_str()?;
        let byte_start = c.read_u64()?;
        let byte_end = c.read_u64()?;
        out.push(OriginalSpan {
            source_uri,
            byte_start,
            byte_end,
        });
    }
    c.finish()?;
    Ok(out)
}

fn read_blob<'a>(c: &mut Cursor<'a>) -> crate::Result<&'a [u8]> {
    let len = c.read_u32()? as usize;
    c.read_bytes(len)
}

/// encode the spans section as an `intpack` repack: a kind byte, the
/// count, a deduped uri pool, then three bitpacked columns (uri index,
/// byte_start, and byte_end - byte_start). lengths are stored as a
/// wrapping difference so reconstruction is byte-exact for any input.
pub fn encode_chunks_original_spans_intpack(spans: &[OriginalSpan]) -> Vec<u8> {
    let mut pool: Vec<&str> = Vec::new();
    let mut pos_of: HashMap<&str, usize> = HashMap::new();
    let mut idx: Vec<u64> = Vec::with_capacity(spans.len());
    let mut starts: Vec<u64> = Vec::with_capacity(spans.len());
    let mut lens: Vec<u64> = Vec::with_capacity(spans.len());
    for s in spans {
        let uri = s.source_uri.as_str();
        // first-appearance index, O(1) via the map. the numbering is
        // identical to a linear pool scan (`pool.len()` at first sight), so
        // the serialized pool + indices stay BYTE-IDENTICAL; this only avoids
        // the O(n*distinct) scan that turns quadratic on many-uri corpora.
        let pos = *pos_of.entry(uri).or_insert_with(|| {
            let p = pool.len();
            pool.push(uri);
            p
        });
        idx.push(pos as u64);
        starts.push(s.byte_start);
        lens.push(s.byte_end.wrapping_sub(s.byte_start));
    }
    let mut out = Vec::new();
    out.push(REPACK_KIND_SPANS);
    out.extend_from_slice(&(spans.len() as u32).to_le_bytes());
    out.extend_from_slice(&(pool.len() as u32).to_le_bytes());
    for uri in &pool {
        out.extend_from_slice(&(uri.len() as u32).to_le_bytes());
        out.extend_from_slice(uri.as_bytes());
    }
    for col in [&idx, &starts, &lens] {
        let blob = pack_u64s(col);
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&blob);
    }
    out
}

/// reconstruct the canonical (raw-encoding) spans payload from the body
/// of an `intpack` repack (the bytes after the kind byte). byte-identical
/// to [`encode_chunks_original_spans`] so `content_hash` is preserved.
pub fn decode_chunks_original_spans_intpack(rest: &[u8]) -> crate::Result<Vec<u8>> {
    let mut c = Cursor::new(rest, SECTION_CHUNKS_ORIGINAL_SPANS);
    let count = c.read_u32()? as usize;
    let n_uris = c.read_u32()? as usize;
    // bound the claim against the bytes before allocating: every pooled uri
    // costs at least its 4-byte length prefix, so a hostile `n_uris` of 1.3G
    // (a real cargo-fuzz finding: a 90-byte payload asked for 31 GB) is a
    // typed error instead of an allocation abort.
    if n_uris > (rest.len() - c.pos) / 4 {
        return Err(c.malformed("spans intpack: uri pool count exceeds payload"));
    }
    let mut pool: Vec<String> = Vec::with_capacity(n_uris);
    for _ in 0..n_uris {
        pool.push(c.read_lp_str()?);
    }
    let idx = unpack_u64s(read_blob(&mut c)?)?;
    let starts = unpack_u64s(read_blob(&mut c)?)?;
    let lens = unpack_u64s(read_blob(&mut c)?)?;
    if idx.len() != count || starts.len() != count || lens.len() != count {
        return Err(c.malformed("spans intpack: column length mismatch"));
    }
    c.finish()?;
    let mut buf = Vec::with_capacity(12 + count * 24);
    write_prefix(&mut buf, count as u64);
    for i in 0..count {
        let uri = pool
            .get(idx[i] as usize)
            .ok_or_else(|| UrnaError::MalformedSectionPayload {
                section_id: SECTION_CHUNKS_ORIGINAL_SPANS,
                reason: "spans intpack: uri index out of range".into(),
            })?;
        write_lp_str(&mut buf, uri)?;
        buf.extend_from_slice(&starts[i].to_le_bytes());
        buf.extend_from_slice(&starts[i].wrapping_add(lens[i]).to_le_bytes());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let spans = vec![
            OriginalSpan {
                source_uri: "doc.txt".into(),
                byte_start: 0,
                byte_end: 10,
            },
            OriginalSpan {
                source_uri: "doc.txt".into(),
                byte_start: 10,
                byte_end: 25,
            },
        ];
        let bytes = encode_chunks_original_spans(&spans).unwrap();
        let back = decode_chunks_original_spans(&bytes, 2).unwrap();
        assert_eq!(spans, back);
    }

    fn corpus(n: u64) -> Vec<OriginalSpan> {
        (0..n)
            .map(|i| OriginalSpan {
                source_uri: "corpus-next/v1.txt".into(), // shared -> deduped
                byte_start: i * 500,
                byte_end: i * 500 + 480,
            })
            .collect()
    }

    #[test]
    fn intpack_decodes_byte_identical_to_raw() {
        let spans = corpus(200);
        let raw = encode_chunks_original_spans(&spans).unwrap();
        let packed = encode_chunks_original_spans_intpack(&spans);
        assert_eq!(packed[0], REPACK_KIND_SPANS);
        assert!(packed.len() < raw.len(), "dedup+intpack must shrink spans");
        let reconstructed = decode_chunks_original_spans_intpack(&packed[1..]).unwrap();
        assert_eq!(reconstructed, raw, "spans repack must rebuild raw bytes");
        assert_eq!(
            decode_chunks_original_spans(&reconstructed, spans.len()).unwrap(),
            spans
        );
    }

    #[test]
    fn intpack_handles_multiple_uris_and_empty() {
        let spans = vec![
            OriginalSpan {
                source_uri: "a.txt".into(),
                byte_start: 5,
                byte_end: 9,
            },
            OriginalSpan {
                source_uri: "b.txt".into(),
                byte_start: 0,
                byte_end: 100,
            },
            OriginalSpan {
                source_uri: "a.txt".into(),
                byte_start: 9,
                byte_end: 12,
            },
        ];
        let raw = encode_chunks_original_spans(&spans).unwrap();
        let packed = encode_chunks_original_spans_intpack(&spans);
        assert_eq!(
            decode_chunks_original_spans_intpack(&packed[1..]).unwrap(),
            raw
        );

        let empty = encode_chunks_original_spans_intpack(&[]);
        assert_eq!(
            decode_chunks_original_spans_intpack(&empty[1..]).unwrap(),
            encode_chunks_original_spans(&[]).unwrap()
        );
    }

    #[test]
    fn intpack_truncated_errors_never_panic() {
        let packed = encode_chunks_original_spans_intpack(&corpus(10));
        for cut in 1..packed.len() {
            let _ = decode_chunks_original_spans_intpack(&packed[1..cut]);
        }
    }
}
