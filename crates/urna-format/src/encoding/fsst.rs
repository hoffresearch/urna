//! fsst (fast static symbol table) text codec over per-chunk canonical
//! streams (the `fsst` wire codec, encoding id 9). a clean-room 255-entry
//! static symbol table maps frequent 1-8 byte substrings to single-byte
//! codes; byte 0xFF is the escape that emits the next raw byte verbatim, so
//! any input round-trips losslessly. the table is built deterministically
//! from a single greedy frequency pass and embedded in the payload header.
//!
//! this is the `TXT_STREAMS_V3` variant: it reuses the txt_streams container
//! (kind byte + count + intpack offset table + N frames, O(1) single-chunk
//! reopen) but each frame is fsst-coded. fsst keeps O(1) single-string decode
//! and wins on SHORT streams where a zstd frame's overhead dominates.
//! [`decode`] rebuilds the EXACT canonical payload byte-for-byte, so
//! `content_hash` is preserved; every read is bounds-checked (typed
//! `UrnaError`, never a panic on a hostile frame).
//!
//! clean-room from the published fsst design (boncz/leis/zukowski, vldb 2020)
//! as surfaced by duckdb's research (255-entry table, 1-8 byte symbols ->
//! 1-byte codes, 0xFF escape). NO code is vendored.

use super::fsst_table::{SymbolTable, parse_table, serialize_table};
use super::intpack::{IntpackReader, pack_u64s};
use super::txt_streams::{build_canonical, malformed, write_container};
use crate::bytes::{le_u32, le_u64};

/// kind/version byte for the fsst-framed variant.
pub const TXT_STREAMS_V3: u8 = 2;

/// escape code: the next byte in the code stream is emitted raw.
const ESCAPE: u8 = 0xFF;

/// encode one string with `table`: greedy longest-match, escape any byte no
/// symbol covers. lossless for arbitrary bytes.
fn encode_one(table: &SymbolTable, input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        match table.longest_match(&input[i..]) {
            Some((code, len)) => {
                out.push(code);
                i += len;
            }
            None => {
                out.push(ESCAPE);
                out.push(input[i]);
                i += 1;
            }
        }
    }
    out
}

/// decode one fsst code stream against the parsed `symbols`. validates that
/// an escape is never the final byte and that codes are in range; never
/// panics on a hostile frame.
fn decode_one(symbols: &[Vec<u8>], codes: &[u8]) -> crate::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(codes.len() * 2);
    let mut i = 0;
    while i < codes.len() {
        let c = codes[i];
        if c == ESCAPE {
            let b = *codes
                .get(i + 1)
                .ok_or_else(|| malformed("fsst: trailing escape"))?;
            out.push(b);
            i += 2;
        } else {
            let sym = symbols
                .get(c as usize)
                .ok_or_else(|| malformed("fsst: code out of table range"))?;
            out.extend_from_slice(sym);
            i += 1;
        }
    }
    Ok(out)
}

/// encode `texts` as per-chunk fsst frames behind the shared txt_streams
/// offset table, with one corpus-wide symbol table embedded after the
/// container header. a pure function of the inputs, so two builds match.
pub fn encode(texts: &[String]) -> crate::Result<Vec<u8>> {
    let corpus: Vec<u8> = texts.iter().flat_map(|t| t.as_bytes().to_vec()).collect();
    let table = SymbolTable::build(&corpus);
    let table_blob = serialize_table(&table);
    let mut streams: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(texts.len() + 1);
    offsets.push(0);
    for t in texts {
        streams.extend_from_slice(&encode_one(&table, t.as_bytes()));
        offsets.push(streams.len() as u64);
    }
    let off_table = pack_u64s(&offsets);
    // the container streams region = u32 table_len + symbol table + frames.
    // the offset table indexes frames relative to the table blob end, so
    // decode splits the region on the stored table length.
    let mut framed = Vec::with_capacity(4 + table_blob.len() + streams.len());
    framed.extend_from_slice(&(table_blob.len() as u32).to_le_bytes());
    framed.extend_from_slice(&table_blob);
    framed.extend_from_slice(&streams);
    Ok(write_container(
        TXT_STREAMS_V3,
        texts.len(),
        &off_table,
        &framed,
    ))
}

/// reconstruct the canonical `chunks_canonical` payload from an fsst-framed
/// `txt_streams` V3 payload. byte-identical to
/// `sections::encode_chunks_canonical`, so `content_hash` is preserved.
pub fn decode(bytes: &[u8]) -> crate::Result<Vec<u8>> {
    let (count, offsets, framed) = parse_v3(bytes)?;
    if framed.len() < 4 {
        return Err(malformed("fsst: truncated region header"));
    }
    let table_len = le_u32(&framed[0..4])? as usize;
    let region = &framed[4..];
    let (symbols, parsed_len) = parse_table(region)?;
    if parsed_len != table_len {
        return Err(malformed("fsst: declared table length mismatch"));
    }
    let streams = region
        .get(table_len..)
        .ok_or_else(|| malformed("fsst: truncated streams region"))?;
    if offsets.last().copied() != Some(streams.len() as u64) {
        return Err(malformed("fsst: final offset != streams length"));
    }
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        let frame = streams
            .get(start..end)
            .ok_or_else(|| malformed("fsst: frame slice out of bounds"))?;
        let raw = decode_one(&symbols, frame)?;
        std::str::from_utf8(&raw).map_err(|e| malformed(format!("fsst: invalid utf-8: {}", e)))?;
        bodies.push(raw);
    }
    build_canonical(count, &bodies)
}

/// parse the V3 container header + intpack offset table, returning the chunk
/// count, the n+1 byte offsets, and the framed region (table + streams).
fn parse_v3(bytes: &[u8]) -> crate::Result<(usize, Vec<u64>, &[u8])> {
    let (kind, rest) = bytes
        .split_first()
        .ok_or_else(|| malformed("fsst: empty"))?;
    if *kind != TXT_STREAMS_V3 {
        return Err(malformed(format!("fsst: unknown kind {}", *kind)));
    }
    if rest.len() < 8 {
        return Err(malformed("fsst: truncated count"));
    }
    let declared = le_u64(&rest[0..8])?;
    let table_bytes = &rest[8..];
    let reader = IntpackReader::parse(table_bytes)?;
    if reader.is_empty() {
        return Err(malformed("fsst: offset table must hold n+1 >= 1"));
    }
    let count = reader.len() - 1;
    if declared != count as u64 {
        return Err(malformed("fsst: declared count != offset count - 1"));
    }
    let mut offsets = Vec::with_capacity(reader.len());
    for i in 0..reader.len() {
        offsets.push(reader.get(i)?);
    }
    if offsets[0] != 0 {
        return Err(malformed("fsst: first offset must be 0"));
    }
    let off_len = pack_u64s(&offsets).len();
    if off_len > table_bytes.len() {
        return Err(malformed("fsst: truncated offset table"));
    }
    let framed = &table_bytes[off_len..];
    for w in offsets.windows(2) {
        if w[1] < w[0] {
            return Err(malformed("fsst: non-monotonic offsets"));
        }
    }
    Ok((count, offsets, framed))
}

// positive + escape-path coverage lives in tests/fsst_roundtrip.rs and the
// negative/fuzz coverage in tests/negative_fsst.rs (both exercise the public
// encode/decode, which drive encode_one/decode_one through every frame).
