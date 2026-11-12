//! libzstd compresses, both decoders decode, the bytes must match: every
//! level the format crate could meet (fast, default, high, its own 19),
//! with and without the content checksum, long and tiny windows, trained
//! and raw-content dictionaries, empty, one-byte, repetitive, random and
//! text-like inputs, frames back to back.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test: a failure is the finding"
)]

use urna_format_parity::{Rng, assert_parity, random_bytes, records, text};
use zstd::zstd_safe::CParameter;

fn inputs() -> Vec<(String, Vec<u8>)> {
    let mut rng = Rng::new(645);
    let mut v = vec![
        ("empty".to_string(), Vec::new()),
        ("one byte".to_string(), vec![b'u']),
        ("repetitive".to_string(), b"urna ".repeat(200_000)),
        ("zeros".to_string(), vec![0u8; 300_000]),
        ("random 64k".to_string(), random_bytes(&mut rng, 65_536)),
        ("random 300k".to_string(), random_bytes(&mut rng, 300_000)),
        ("text 1m".to_string(), text(&mut rng, 1_000_000)),
    ];
    let mut mixed = text(&mut rng, 150_000);
    for _ in 0..200 {
        let at = rng.below(mixed.len());
        let n = 1 + rng.below(300);
        let noise = random_bytes(&mut rng, n);
        mixed.splice(at..at, noise);
    }
    v.push(("text with noise".to_string(), mixed));
    for n in [2usize, 7, 31, 100, 257, 1000, 4097] {
        v.push((format!("text {n}"), text(&mut rng, n)));
    }
    v
}

fn compress(data: &[u8], level: i32, checksum: bool, dict: Option<&[u8]>) -> Vec<u8> {
    let mut c = match dict {
        Some(dict) => zstd::bulk::Compressor::with_dictionary(level, dict).unwrap(),
        None => zstd::bulk::Compressor::new(level).unwrap(),
    };
    c.set_parameter(CParameter::ChecksumFlag(checksum)).unwrap();
    c.compress(data).unwrap()
}

#[test]
fn levels_and_checksums() {
    for (name, data) in inputs() {
        for level in [-5, 1, 3, 9, 19] {
            for checksum in [false, true] {
                let frame = compress(&data, level, checksum, None);
                let label = format!("{name} level {level} checksum {checksum}");
                assert_parity(&label, Some(&data), &frame, None);
            }
        }
    }
}

/// streamed frames carry a window descriptor and no content size; a long
/// window with long-distance matching reaches back past 8 MiB, a 1 KiB
/// window makes every block small.
#[test]
fn streamed_windows() {
    let mut rng = Rng::new(8878);
    let block = random_bytes(&mut rng, 9 << 20);
    let mut far = block.clone();
    far.extend_from_slice(&text(&mut rng, 1 << 16));
    far.extend_from_slice(&block);
    let tiny = text(&mut rng, 300_000);
    let tiny_fast = text(&mut rng, 100_000);
    let cases: [(&str, &[u8], i32, u32, bool); 4] = [
        ("long window ldm", &far, 3, 27, true),
        ("long window", &far, 1, 25, false),
        ("tiny window", &tiny, 19, 10, false),
        ("tiny window fast", &tiny_fast, 1, 10, false),
    ];
    for (name, data, level, window, ldm) in cases {
        let mut enc = zstd::stream::Encoder::new(Vec::new(), level).unwrap();
        enc.window_log(window).unwrap();
        enc.long_distance_matching(ldm).unwrap();
        enc.include_checksum(true).unwrap();
        std::io::Write::write_all(&mut enc, data).unwrap();
        let frame = enc.finish().unwrap();
        assert_parity(name, Some(data), &frame, None);
    }
}

#[test]
fn frames_back_to_back() {
    let mut rng = Rng::new(3);
    let a = text(&mut rng, 50_000);
    let b = random_bytes(&mut rng, 5_000);
    let mut frames = compress(&a, 19, true, None);
    frames.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 4, 0, 0, 0, 1, 2, 3, 4]);
    frames.extend_from_slice(&compress(&b, 3, false, None));
    frames.extend_from_slice(&compress(b"", 1, false, None));
    frames.extend_from_slice(&compress(&a, 1, false, None));
    let mut want = a.clone();
    want.extend_from_slice(&b);
    want.extend_from_slice(&a);
    assert_parity("back to back", Some(&want), &frames, None);
}

/// a dictionary trained the way the writer trains it (zdict over sorted
/// unique records), frames with and without its ID, small records and a
/// large multi-block input; then raw content as the dictionary.
#[test]
fn dictionaries() {
    let mut rng = Rng::new(5);
    let mut samples = records(&mut rng, 3_000);
    samples.sort_unstable();
    samples.dedup();
    let data: Vec<u8> = samples.iter().flat_map(|s| s.bytes()).collect();
    let sizes: Vec<usize> = samples.iter().map(String::len).collect();
    let dict = zstd::dict::from_continuous(&data, &sizes, 16 * 1024).unwrap();
    let large = text(&mut rng, 400_000);
    for level in [1, 3, 9, 19] {
        for (i, s) in samples.iter().enumerate().step_by(7) {
            let frame = compress(s.as_bytes(), level, i % 2 == 0, Some(&dict));
            assert_parity(
                &format!("record {i} level {level}"),
                Some(s.as_bytes()),
                &frame,
                Some(&dict),
            );
        }
        let frame = compress(&large, level, true, Some(&dict));
        assert_parity(
            &format!("large level {level}"),
            Some(&large),
            &frame,
            Some(&dict),
        );
        let mut c = zstd::bulk::Compressor::with_dictionary(level, &dict).unwrap();
        c.set_parameter(CParameter::DictIdFlag(false)).unwrap();
        let frame = c.compress(samples[0].as_bytes()).unwrap();
        assert_parity(
            "no dict id",
            Some(samples[0].as_bytes()),
            &frame,
            Some(&dict),
        );
    }
    // urna's own trainer and dict-framed streams, stream by stream.
    let texts = records(&mut rng, 2_000);
    let mut unique = texts.clone();
    unique.sort_unstable();
    unique.dedup();
    let dict = urna_format::encoding::train_dict(&unique).expect("trains");
    let framed = urna_format::encoding::encode_zstd_dict(&texts, &dict).unwrap();
    for (i, stream) in urna_format_parity::container_streams(&framed)
        .iter()
        .enumerate()
    {
        assert_parity(
            &format!("urna stream {i}"),
            Some(texts[i].as_bytes()),
            stream,
            Some(&dict),
        );
    }
    // raw content: no magic, the bytes are the history.
    let raw = text(&mut rng, 8_192);
    for level in [1, 19] {
        let s = text(&mut rng, 3_000);
        let frame = compress(&s, level, true, Some(&raw));
        assert_parity(
            &format!("raw dict level {level}"),
            Some(&s),
            &frame,
            Some(&raw),
        );
    }
}
