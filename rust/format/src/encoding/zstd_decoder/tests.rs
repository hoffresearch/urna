//! hand-made frames: a few well-formed ones that exercise each block type
//! without the encoder, and a regression corpus of malformed ones that must
//! each end in an error (never a panic, never past the limit). the parity
//! workspace covers what the real encoder writes.

use super::ZstdDecoder;
use super::xxhash::xxh64;

const MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
const LIMIT: usize = 1 << 20;

fn decode(src: &[u8]) -> Result<Vec<u8>, super::ZstdError> {
    ZstdDecoder::default().decompress(src, LIMIT)
}

/// a frame: magic, descriptor, the given header fields, then the blocks.
fn frame(descriptor: u8, fields: &[u8], blocks: &[u8]) -> Vec<u8> {
    let mut f = MAGIC.to_vec();
    f.push(descriptor);
    f.extend_from_slice(fields);
    f.extend_from_slice(blocks);
    f
}

/// a block header: last flag, type, size.
fn block(last: bool, kind: u32, size: u32) -> [u8; 3] {
    let h = (size << 3) | (kind << 1) | u32::from(last);
    let [a, b, c, _] = h.to_le_bytes();
    [a, b, c]
}

fn raw_hello() -> Vec<u8> {
    let mut blocks = block(true, 0, 5).to_vec();
    blocks.extend_from_slice(b"hello");
    // single segment, 1-byte content size 5.
    frame(0x20, &[5], &blocks)
}

/// one compressed block: raw literals "abc", one sequence with the
/// predefined tables (literals length 3, offset value 6 = offset 3, match
/// length 6). states LL 3, OF 14, ML 3, then 2 extra offset bits = 2.
fn one_sequence(stream: [u8; 3]) -> Vec<u8> {
    let body = [
        0x18, b'a', b'b', b'c', 0x01, 0x00, stream[0], stream[1], stream[2],
    ];
    let mut blocks = block(true, 2, body.len() as u32).to_vec();
    blocks.extend_from_slice(&body);
    // window descriptor 0 (1 KiB), no content size.
    frame(0x00, &[0x00], &blocks)
}

const SEQ_STREAM: [u8; 3] = [0x0E, 0x6E, 0x08];

#[test]
fn raw_rle_and_empty_frames() {
    assert_eq!(decode(&raw_hello()).unwrap(), b"hello");
    let mut rle = block(true, 1, 10).to_vec();
    rle.push(b'a');
    assert_eq!(decode(&frame(0x20, &[10], &rle)).unwrap(), b"aaaaaaaaaa");
    // what libzstd writes for an empty input: content size 0, one empty
    // raw block.
    assert_eq!(decode(&frame(0x20, &[0], &block(true, 0, 0))).unwrap(), b"");
    // no frame at all is no output, as libzstd reads it.
    assert_eq!(decode(&[]).unwrap(), b"");
}

#[test]
fn compressed_block_with_one_sequence() {
    assert_eq!(decode(&one_sequence(SEQ_STREAM)).unwrap(), b"abcabcabc");
    // literals only: raw "hello", zero sequences.
    let body = [0x28, b'h', b'e', b'l', b'l', b'o', 0x00];
    let mut blocks = block(true, 2, body.len() as u32).to_vec();
    blocks.extend_from_slice(&body);
    assert_eq!(decode(&frame(0x00, &[0x00], &blocks)).unwrap(), b"hello");
}

#[test]
fn frames_back_to_back_and_skippable() {
    let mut src = raw_hello();
    src.extend_from_slice(&[0x5A, 0x2A, 0x4D, 0x18, 3, 0, 0, 0, b'x', b'y', b'z']);
    src.extend_from_slice(&one_sequence(SEQ_STREAM));
    assert_eq!(decode(&src).unwrap(), b"helloabcabcabc");
}

#[test]
fn content_checksum() {
    let mut blocks = block(true, 0, 5).to_vec();
    blocks.extend_from_slice(b"hello");
    let sum = (xxh64(b"hello", 0) as u32).to_le_bytes();
    let mut good = blocks.clone();
    good.extend_from_slice(&sum);
    assert_eq!(decode(&frame(0x24, &[5], &good)).unwrap(), b"hello");
    let mut bad = blocks.clone();
    bad.extend_from_slice(&[sum[0] ^ 1, sum[1], sum[2], sum[3]]);
    assert!(decode(&frame(0x24, &[5], &bad)).is_err());
    assert!(decode(&frame(0x24, &[5], &blocks)).is_err()); // checksum missing
}

#[test]
fn raw_content_dictionary_is_history() {
    // literals "x", one sequence: literals length 1, offset 4 (value 7:
    // offset code 2 with extra bits 3), match length 3: it reaches 3 bytes
    // back into the dictionary "WXYZ" past the one literal.
    // states as in `one_sequence` but LL code 1 (state 2), ML code 0
    // (state 0): bits = 1<<19 | 2<<13 | 14<<8 | 0<<2 | 3.
    let v: u32 = (1 << 19) | (2 << 13) | (14 << 8) | 3;
    let [s0, s1, s2, _] = v.to_le_bytes();
    let body = [0x08, b'x', 0x01, 0x00, s0, s1, s2];
    let mut blocks = block(true, 2, body.len() as u32).to_vec();
    blocks.extend_from_slice(&body);
    let src = frame(0x00, &[0x00], &blocks);
    let mut dec = ZstdDecoder::with_dictionary(b"WXYZ").unwrap();
    assert_eq!(dec.decompress(&src, LIMIT).unwrap(), b"xXYZ");
    // without the dictionary the offset points before the start.
    assert!(decode(&src).is_err());
}

#[test]
fn malformed_frames_are_errors() {
    let hello = raw_hello();
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("short", MAGIC[..3].to_vec()),
        ("bad magic", vec![0, 1, 2, 3, 4, 5, 6, 7]),
        ("reserved bit", frame(0x28, &[5], &hello[6..])),
        ("window too large", frame(0x00, &[0xF8], &block(true, 0, 0))),
        ("no block", frame(0x20, &[5], &[])),
        ("reserved block type", frame(0x20, &[0], &block(true, 3, 0))),
        ("raw truncated", hello[..hello.len() - 1].to_vec()),
        ("content size mismatch", frame(0x20, &[6], &hello[6..])),
        (
            "needs a dictionary",
            frame(0x01, &[0x00, 7], &block(true, 0, 0)),
        ),
        ("compressed too small", {
            let mut b = block(true, 2, 1).to_vec();
            b.push(0);
            frame(0x00, &[0x00], &b)
        }),
    ];
    let mut trailing = hello.clone();
    trailing.extend_from_slice(&[1, 2, 3]);
    cases.push(("trailing bytes", trailing));
    let mut skip = hello.clone();
    skip.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 9, 0, 0, 0, 1]);
    cases.push(("skippable truncated", skip));
    // the sequence frame, broken one field at a time.
    let seq = one_sequence(SEQ_STREAM);
    let at = seq.len() - 3; // start of the sequences stream
    let mut repeat = seq.clone();
    repeat[at - 1] = 0xC0; // literals length in repeat mode, nothing to repeat
    cases.push(("repeat without a table", repeat));
    let mut reserved = seq.clone();
    reserved[at - 1] = 0x01;
    cases.push(("modes reserved bits", reserved));
    let mut far = seq.clone();
    far[at] |= 1; // offset extra bits 3: offset 4, before the frame start
    cases.push(("offset before the start", far));
    let mut no_marker = seq.clone();
    no_marker[seq.len() - 1] = 0;
    cases.push(("stream without its marker", no_marker));
    let mut longer = seq.clone();
    longer[seq.len() - 1] = 0x10; // one more bit than the sequence reads
    cases.push(("stream not consumed", longer));
    // raw literals "ab", but the sequence copies 3 of them.
    let body = [
        0x10,
        b'a',
        b'b',
        0x01,
        0x00,
        SEQ_STREAM[0],
        SEQ_STREAM[1],
        SEQ_STREAM[2],
    ];
    let mut blocks = block(true, 2, body.len() as u32).to_vec();
    blocks.extend_from_slice(&body);
    cases.push(("too few literals", frame(0x00, &[0x00], &blocks)));
    for (name, src) in cases {
        assert!(decode(&src).is_err(), "{name} decoded");
    }
}

#[test]
fn limit_is_a_hard_bound() {
    // declared bigger than the limit: refused before any allocation.
    let fcs = (LIMIT as u32 + 1).to_le_bytes();
    let mut rle = block(true, 1, 4).to_vec();
    rle.push(b'a');
    assert!(decode(&frame(0xA0, &fcs, &rle)).is_err());
    // no declared size: an RLE block past the limit stops at the limit.
    let mut big = block(true, 1, (1 << 21) - 1).to_vec();
    big.push(b'a');
    let mut dec = ZstdDecoder::default();
    assert!(dec.decompress(&frame(0x00, &[0x58], &big), 1000).is_err());
    assert_eq!(dec.decompress(&raw_hello(), 5).unwrap(), b"hello");
    assert!(dec.decompress(&raw_hello(), 4).is_err());
}
