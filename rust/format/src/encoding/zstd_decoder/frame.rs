//! frames and blocks, RFC 8878 sections 3.1.1 and 3.1.2: the frame header
//! (descriptor, window, dictionary ID, content size), skippable frames,
//! raw / RLE / compressed blocks, the content checksum, and any number of
//! frames back to back, each decoded on its own.

use super::dictionary::Dictionary;
use super::huffman::HuffTable;
use super::literals::{self, TreeSource, Trees};
use super::sequences::{self, Output, SeqState};
use super::xxhash::xxh64;
use super::{Res, corrupt, make_room};

const FRAME_MAGIC: u32 = 0xFD2F_B528;
const SKIPPABLE_MAGIC: u32 = 0x184D_2A50;
const SKIPPABLE_MASK: u32 = 0xFFFF_FFF0;
/// the largest block, and the largest window log libzstd accepts.
const BLOCK_MAX: u64 = 128 * 1024;
const WINDOW_LOG_MAX: u32 = 31;

/// the buffers and state one decoder reuses from frame to frame.
#[derive(Debug, Default)]
pub(super) struct Scratch {
    lits: Vec<u8>,
    huffman: HuffTable,
    tree: TreeSource,
    seq: SeqState,
}

/// what a frame header says.
#[derive(Debug, Clone, Copy)]
struct Header {
    len: usize,
    window: u64,
    dict_id: u32,
    content_size: Option<u64>,
    checksum: bool,
}

fn le(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | u64::from(b))
}

fn le32_at(src: &[u8], at: usize) -> Res<u32> {
    let w = src.get(at..at + 4).ok_or(corrupt("truncated frame"))?;
    Ok(u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
}

/// parse the header after the magic number; `src` starts at the
/// descriptor byte.
fn read_header(src: &[u8]) -> Res<Header> {
    let fhd = *src.first().ok_or(corrupt("frame header truncated"))?;
    if fhd & 0x08 != 0 {
        return Err(corrupt("frame header reserved bit set"));
    }
    let single_segment = fhd & 0x20 != 0;
    let did_len = [0usize, 1, 2, 4][usize::from(fhd & 3)];
    let fcs_len = match fhd >> 6 {
        0 => usize::from(single_segment),
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let len = 1 + usize::from(!single_segment) + did_len + fcs_len;
    let fields = src.get(..len).ok_or(corrupt("frame header truncated"))?;
    let mut pos = 1;
    let mut window = 0u64;
    if !single_segment {
        let wd = fields[pos];
        pos += 1;
        let log = 10 + u32::from(wd >> 3);
        if log > WINDOW_LOG_MAX {
            return Err(corrupt("frame window too large"));
        }
        let base = 1u64 << log;
        window = base + (base >> 3) * u64::from(wd & 7);
    }
    let dict_id = le(&fields[pos..pos + did_len]) as u32;
    pos += did_len;
    let content_size = match fcs_len {
        0 => None,
        2 => Some(le(&fields[pos..pos + 2]) + 256),
        n => Some(le(&fields[pos..pos + n])),
    };
    if single_segment {
        window = content_size.unwrap_or(0);
    }
    Ok(Header {
        len,
        window,
        dict_id,
        content_size,
        checksum: fhd & 0x04 != 0,
    })
}

/// decode every frame of `src` into `out`, never growing it past `limit`
/// bytes. an empty input is zero frames, as libzstd reads it.
pub(super) fn decode_all(
    src: &[u8],
    dict: Option<&Dictionary<'_>>,
    limit: usize,
    out: &mut Vec<u8>,
    scratch: &mut Scratch,
) -> Res<()> {
    let mut rest = src;
    while !rest.is_empty() {
        if rest.len() < 5 {
            return Err(corrupt("trailing bytes after the last frame"));
        }
        let magic = le32_at(rest, 0)?;
        if magic & SKIPPABLE_MASK == SKIPPABLE_MAGIC {
            let size = le32_at(rest, 4)? as usize;
            let end = size
                .checked_add(8)
                .filter(|&end| end <= rest.len())
                .ok_or(corrupt("skippable frame truncated"))?;
            rest = &rest[end..];
            continue;
        }
        if magic != FRAME_MAGIC {
            return Err(corrupt("unknown frame magic"));
        }
        let used = decode_frame(&rest[4..], dict, limit, out, scratch)?;
        rest = &rest[4 + used..];
    }
    Ok(())
}

/// decode one frame (after its magic); returns the bytes it took.
fn decode_frame(
    src: &[u8],
    dict: Option<&Dictionary<'_>>,
    limit: usize,
    out: &mut Vec<u8>,
    scratch: &mut Scratch,
) -> Res<usize> {
    let header = read_header(src)?;
    if src.len() < header.len + 3 {
        return Err(corrupt("frame without a block"));
    }
    if header.dict_id != 0 && Some(header.dict_id) != dict.map(|d| d.id) {
        return Err(corrupt("frame needs another dictionary"));
    }
    let frame_start = out.len();
    let block_max = header.window.min(BLOCK_MAX) as usize;
    let mut pos = header.len;
    if let Some(size) = header.content_size {
        let room = limit.saturating_sub(frame_start);
        if size > room as u64 {
            return Err(corrupt("frame declares more bytes than the limit"));
        }
        // trust the declared size only as far as the input could produce:
        // every block costs at least 4 bytes and yields at most block_max.
        let plausible = (src.len() / 4 + 1).saturating_mul(block_max.max(1));
        make_room(out, (size as usize).min(plausible), limit)?;
    }
    let entropy = dict.and_then(|d| d.entropy.as_ref());
    scratch.tree = if entropy.is_some() {
        TreeSource::Dictionary
    } else {
        TreeSource::Missing
    };
    scratch.seq.reset(entropy.map(|e| (&e.tables, e.reps)));
    let history = dict.map_or(&[][..], |d| d.history);
    loop {
        let bh = src
            .get(pos..pos + 3)
            .ok_or(corrupt("block header truncated"))?;
        let bh = le(bh) as usize;
        pos += 3;
        let (last, kind, size) = (bh & 1 == 1, (bh >> 1) & 3, bh >> 3);
        match kind {
            0 => {
                let raw = src
                    .get(pos..pos + size)
                    .ok_or(corrupt("raw block truncated"))?;
                make_room(out, size, limit)?;
                out.extend_from_slice(raw);
                pos += size;
            }
            1 => {
                let byte = *src.get(pos).ok_or(corrupt("rle block truncated"))?;
                make_room(out, size, limit)?;
                out.resize(out.len() + size, byte);
                pos += 1;
            }
            2 => {
                if size > block_max {
                    return Err(corrupt("compressed block larger than the block maximum"));
                }
                let body = src
                    .get(pos..pos + size)
                    .ok_or(corrupt("compressed block truncated"))?;
                let mut output = Output {
                    out: &mut *out,
                    frame_start,
                    limit,
                    history,
                };
                decode_block(body, block_max, dict, scratch, &mut output)?;
                pos += size;
            }
            _ => return Err(corrupt("reserved block type")),
        }
        if last {
            break;
        }
    }
    let produced = (out.len() - frame_start) as u64;
    if header.content_size.is_some_and(|size| size != produced) {
        return Err(corrupt("frame content size does not match its blocks"));
    }
    if header.checksum {
        let stored = le32_at(src, pos).map_err(|_| corrupt("frame checksum truncated"))?;
        if xxh64(&out[frame_start..], 0) as u32 != stored {
            return Err(corrupt("frame checksum mismatch"));
        }
        pos += 4;
    }
    Ok(pos)
}

/// a compressed block: the literals section, then the sequences section
/// that executes them.
fn decode_block(
    body: &[u8],
    block_max: usize,
    dict: Option<&Dictionary<'_>>,
    scratch: &mut Scratch,
    output: &mut Output<'_, '_>,
) -> Res<()> {
    if body.len() < 2 {
        return Err(corrupt("compressed block too small"));
    }
    let entropy = dict.and_then(|d| d.entropy.as_ref());
    let mut trees = Trees {
        source: scratch.tree,
        frame: &mut scratch.huffman,
        dictionary: entropy.map(|e| &e.huffman),
    };
    let used = literals::decode(body, block_max, &mut trees, &mut scratch.lits)?;
    scratch.tree = trees.source;
    let seq = body.get(used..).ok_or(corrupt("literals past the block"))?;
    sequences::decode(
        seq,
        &mut scratch.seq,
        entropy.map(|e| &e.tables),
        &scratch.lits,
        output,
    )
}
