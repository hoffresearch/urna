//! property tests for the section codecs.
//!
//! two claims per codec, the ones the citation uri and the reproducible
//! build depend on and that the fixed-input tests only sample:
//!
//! 1. `decode(encode(x)) == x`, byte-identical to the raw payload the codec
//!    replaces (so `content_hash` cannot move);
//! 2. `encode(x) == encode(x)`, the encoder is a pure function of its input
//!    (so two builds of the same corpus match).
//!
//! the inputs come from a seeded splitmix64 (`Gen`), so every run draws
//! the same cases: 256 per property, `URNA_PROP_CASES=5000` for a local
//! soak. a failing case names its seed; `URNA_PROP_SEED=<seed>` reruns that
//! one case alone.

// not under miri: property volume is not what miri is for: hundreds of cases per property, each a full encode + decode; the fixed-input codec tests cover the same paths under miri.
#![cfg(not(miri))]

use urna_format::encoding::{IntpackReader, pack_u64s, unpack_u64s};
use urna_format::encoding::{
    decode_dedup_map, decode_fsst_payload, decode_txt_streams_payload, decode_zstd_dict_payload,
    dedup, encode_dedup_map, encode_fsst, encode_zstd_dict, expand_dedup, train_dict,
};
use urna_format::{
    INT4_BLOCK, Int4EmbeddingsView, Int8EmbeddingsView, TxtStreams, decode_payload,
    encode_chunks_canonical, encode_int4_embeddings, encode_int8_embeddings, encode_txt_streams,
    expected_embeddings_size, f16_bytes_to_f32, f32_to_f16_bytes, nibble_to_i4, pack_nibbles,
};
use urna_format::{
    SECTION_ENCODING_FSST, SECTION_ENCODING_INT4, SECTION_ENCODING_INT8, SECTION_ENCODING_INTPACK,
    SECTION_ENCODING_RAW, SECTION_ENCODING_TXT_STREAMS,
};

// ---- the generator -------------------------------------------------------

/// splitmix64 and the draws the properties need.
struct Gen(u64);

impl Gen {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// uniform in `lo..=hi`.
    fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.u64() % (hi - lo + 1) as u64) as usize
    }

    /// uniform in `[lo, hi)`.
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * (hi - lo)
    }

    /// the index of a weight, drawn in proportion to it.
    fn weighted(&mut self, weights: &[u64]) -> usize {
        let mut r = self.u64() % weights.iter().sum::<u64>();
        for (i, &w) in weights.iter().enumerate() {
            if r < w {
                return i;
            }
            r -= w;
        }
        weights.len() - 1
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.int(0, items.len() - 1)]
    }

    /// `lo..=hi` characters drawn from `alphabet`.
    fn string(&mut self, alphabet: &[char], lo: usize, hi: usize) -> String {
        let n = self.int(lo, hi);
        (0..n).map(|_| *self.pick(alphabet)).collect()
    }
}

/// `body` over the property's cases, each from its own seed; a failure
/// names the seed that reproduces it.
fn property(name: &str, body: impl Fn(&mut Gen)) {
    let one = std::env::var("URNA_PROP_SEED")
        .ok()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    let cases: u64 = std::env::var("URNA_PROP_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let base = name.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01B3)
    });
    let seeds: Vec<u64> = match one {
        Some(seed) => vec![seed],
        None => (0..cases)
            .map(|c| base ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect(),
    };
    for seed in seeds {
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&mut Gen(seed))));
        if run.is_err() {
            panic!("{name} failed on seed {seed:#x}; rerun it alone with URNA_PROP_SEED={seed:#x}");
        }
    }
}

const LOWER: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z',
];
const HEX: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];
/// letters and digits of one, two, three and four utf-8 bytes, and a space.
const UNICODE: &[char] = &[
    'a', 'Z', '7', ' ', 'é', 'ß', 'ñ', 'ø', 'Ж', 'я', 'λ', 'Ω', 'ع', '٣', 'א', 'अ', '中', '文',
    'ア', '한', '一', '𝔘', '𐍈', '𝟙',
];

fn with(base: &[char], extra: &[char]) -> Vec<char> {
    base.iter().chain(extra).copied().collect()
}

// ---- inputs --------------------------------------------------------------

/// integers across every intpack frame width: tiny, u32-range, full u64.
fn u64s(g: &mut Gen) -> Vec<u64> {
    let n = g.int(0, 1999);
    (0..n)
        .map(|_| match g.weighted(&[4, 3, 1]) {
            0 => g.u64() % 16,
            1 => g.u64() % u32::MAX as u64,
            _ => g.u64(),
        })
        .collect()
}

/// canonical texts: empty, ascii, multibyte utf-8, long runs; 0..200 chunks.
fn texts(g: &mut Gen) -> Vec<String> {
    let ascii = with(LOWER, &[' ', ',', '.']);
    let n = g.int(0, 199);
    (0..n)
        .map(|_| match g.weighted(&[1, 4, 2, 1]) {
            0 => String::new(),
            1 => g.string(&ascii, 1, 120),
            2 => g.string(UNICODE, 1, 80),
            _ => {
                let mut s = String::new();
                for _ in 0..g.int(1, 6) {
                    s.push_str(g.pick(&["the quick brown fox ", "lorem ipsum dolor ", "urna://"]));
                }
                s + &g.string(HEX, 0, 64)
            }
        })
        .collect()
}

/// texts with duplicates guaranteed: every entry is drawn from a pool of at
/// most 12 distinct strings.
fn texts_with_duplicates(g: &mut Gen) -> Vec<String> {
    let alphabet = with(LOWER, &[' ']);
    let pool: Vec<String> = (0..g.int(1, 11))
        .map(|_| g.string(&alphabet, 0, 40))
        .collect();
    (0..g.int(0, 299)).map(|_| g.pick(&pool).clone()).collect()
}

/// enough boilerplate-rich material for `train_dict` to accept: the trainer
/// needs at least 8 unique samples and some shared substrings.
fn dict_corpus(g: &mut Gen) -> Vec<String> {
    let tail = with(
        LOWER,
        &['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' '],
    );
    let phrases = [
        "patient presents with ",
        "dose adjusted to ",
        "follow-up in ",
        "see section ",
    ];
    (0..g.int(16, 79))
        .map(|_| {
            let mut s = String::new();
            for _ in 0..g.int(1, 4) {
                s.push_str(g.pick(&phrases));
            }
            s + &g.string(&tail, 8, 60)
        })
        .collect()
}

/// `n` (1..24) l2-normalized rows of `dim` finite f32, none of them zero.
fn normalized_rows(g: &mut Gen, dim: usize) -> (Vec<f32>, usize) {
    let n = g.int(1, 23);
    let mut v = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut row: Vec<f32> = (0..dim).map(|_| g.f32(-1.0, 1.0)).collect();
        if row.iter().all(|x| *x == 0.0) {
            row[0] = 1.0;
        }
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.extend(row.iter().map(|x| x / norm));
    }
    (v, n)
}

#[allow(clippy::expect_used, reason = "a test helper, outside #[test]")]
fn raw_canonical(texts: &[String]) -> Vec<u8> {
    encode_chunks_canonical(texts).expect("raw canonical encoding never fails")
}

// ---- intpack -----------------------------------------------------------

#[test]
fn intpack_roundtrips_and_is_byte_stable() {
    property("intpack_roundtrips_and_is_byte_stable", |g| {
        let values = u64s(g);
        let packed = pack_u64s(&values);
        assert_eq!(
            pack_u64s(&values),
            packed.clone(),
            "encoder is not a pure function"
        );
        assert_eq!(unpack_u64s(&packed).expect("unpack"), values.clone());
        let reader = IntpackReader::parse(&packed).expect("parse");
        assert_eq!(reader.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            assert_eq!(reader.get(i).expect("in-range get"), *v);
        }
        assert!(
            reader.get(values.len()).is_err(),
            "past-the-end get must be a typed error"
        );
    });
}

// ---- txt_streams / fsst / zstd_dict: every text codec decodes to the raw
// ---- canonical payload, byte for byte -----------------------------------

#[test]
fn txt_streams_roundtrips_and_is_byte_stable() {
    property("txt_streams_roundtrips_and_is_byte_stable", |g| {
        let texts = texts(g);
        let raw = raw_canonical(&texts);
        let enc = encode_txt_streams(&texts).expect("encode");
        assert_eq!(
            encode_txt_streams(&texts).expect("encode twice"),
            enc.clone()
        );
        assert_eq!(
            decode_txt_streams_payload(&enc).expect("decode"),
            raw.clone()
        );
        assert_eq!(
            decode_payload(SECTION_ENCODING_TXT_STREAMS, &enc)
                .expect("dispatch")
                .into_owned(),
            raw
        );
        let view = TxtStreams::parse(&enc).expect("parse");
        assert_eq!(view.len(), texts.len());
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(&view.text(i).expect("text"), t);
        }
    });
}

#[test]
fn fsst_roundtrips_and_is_byte_stable() {
    property("fsst_roundtrips_and_is_byte_stable", |g| {
        let texts = texts(g);
        let raw = raw_canonical(&texts);
        let enc = encode_fsst(&texts).expect("encode");
        assert_eq!(encode_fsst(&texts).expect("encode twice"), enc.clone());
        assert_eq!(decode_fsst_payload(&enc).expect("decode"), raw.clone());
        assert_eq!(
            decode_payload(SECTION_ENCODING_FSST, &enc)
                .expect("dispatch")
                .into_owned(),
            raw
        );
    });
}

#[test]
fn zstd_dict_roundtrips_and_is_byte_stable() {
    property("zstd_dict_roundtrips_and_is_byte_stable", |g| {
        let texts = dict_corpus(g);
        let mut unique = texts.clone();
        unique.sort();
        unique.dedup();
        if unique.len() < 8 {
            return;
        }
        let Some(dict) = train_dict(&unique) else {
            return;
        };
        assert_eq!(
            train_dict(&unique).expect("train twice"),
            dict.clone(),
            "dictionary training is not deterministic"
        );
        let raw = raw_canonical(&texts);
        let enc = encode_zstd_dict(&texts, &dict).expect("encode");
        assert_eq!(
            encode_zstd_dict(&texts, &dict).expect("encode twice"),
            enc.clone()
        );
        assert_eq!(decode_zstd_dict_payload(&enc, &dict).expect("decode"), raw);
    });
}

// ---- dedup ---------------------------------------------------------------

#[test]
fn dedup_map_roundtrips_and_expands_exactly() {
    property("dedup_map_roundtrips_and_expands_exactly", |g| {
        let texts = texts_with_duplicates(g);
        let d = dedup(&texts);
        let map = encode_dedup_map(&d.back_refs);
        assert_eq!(encode_dedup_map(&d.back_refs), map.clone());
        let refs = decode_dedup_map(&map).expect("decode map");
        assert_eq!(&refs, &d.back_refs);
        assert_eq!(
            expand_dedup(&d.unique, &refs).expect("expand"),
            texts.clone()
        );
        // the unique pool is first-seen order with no repeats.
        let mut seen = std::collections::HashSet::new();
        assert!(d.unique.iter().all(|u| seen.insert(u.clone())));
        assert!(d.unique.len() <= texts.len());
    });
}

// ---- embeddings: float16 / int8 / int4 -----------------------------------

#[test]
fn float16_roundtrips_within_one_ulp() {
    property("float16_roundtrips_within_one_ulp", |g| {
        let values: Vec<f32> = (0..g.int(0, 2047)).map(|_| g.f32(-1.0, 1.0)).collect();
        let bytes = f32_to_f16_bytes(&values);
        assert_eq!(f32_to_f16_bytes(&values), bytes.clone());
        assert_eq!(bytes.len(), values.len() * 2);
        let back = f16_bytes_to_f32(&bytes);
        for (x, y) in values.iter().zip(back.iter()) {
            // f16 has 11 significand bits: relative error <= 2^-11 (+ a
            // subnormal floor near zero).
            let tol = x.abs() * (1.0 / 2048.0) + 1e-6;
            assert!(
                (x - y).abs() <= tol,
                "{x} -> {y} exceeds f16 tolerance {tol}"
            );
        }
    });
}

#[test]
fn int8_payload_is_sized_stable_and_reconstructs_within_half_a_step() {
    property(
        "int8_payload_is_sized_stable_and_reconstructs_within_half_a_step",
        |g| {
            let dim = g.int(1, 511);
            let (rows, n) = normalized_rows(g, dim);
            let enc = encode_int8_embeddings(&rows, n, dim).expect("encode");
            assert_eq!(
                encode_int8_embeddings(&rows, n, dim).expect("encode twice"),
                enc.clone()
            );
            assert_eq!(Some(enc.len()), expected_embeddings_size("int8", n, dim));
            assert_eq!(
                decode_payload(SECTION_ENCODING_INT8, &enc)
                    .expect("dispatch")
                    .len(),
                enc.len(),
                "embedding encodings are their own canonical bytes"
            );
            let view = Int8EmbeddingsView::parse(&enc, n, dim).expect("parse");
            assert_eq!((view.n, view.dim), (n, dim));
            for i in 0..n {
                let scale = view.scale(i);
                assert!(scale.is_finite() && scale > 0.0);
                let row = view.row_bytes(i);
                for (j, code) in row.iter().enumerate() {
                    let x = rows[i * dim + j];
                    let y = *code as i8 as f32 * scale;
                    assert!(
                        (x - y).abs() <= scale * 0.5 + 1e-5,
                        "row {i} col {j}: {x} vs {y} (scale {scale})"
                    );
                }
            }
        },
    );
}

#[test]
fn int4_payload_is_sized_stable_and_reconstructs_within_half_a_step() {
    property(
        "int4_payload_is_sized_stable_and_reconstructs_within_half_a_step",
        |g| {
            let dim = g.int(1, 8) * INT4_BLOCK;
            let (rows, n) = normalized_rows(g, dim);
            let enc = encode_int4_embeddings(&rows, n, dim).expect("encode");
            assert_eq!(
                encode_int4_embeddings(&rows, n, dim).expect("encode twice"),
                enc.clone()
            );
            assert_eq!(Some(enc.len()), expected_embeddings_size("int4", n, dim));
            assert_eq!(
                decode_payload(SECTION_ENCODING_INT4, &enc)
                    .expect("dispatch")
                    .len(),
                enc.len()
            );
            let view = Int4EmbeddingsView::parse(&enc, n, dim).expect("parse");
            assert_eq!((view.n, view.dim, view.blocks), (n, dim, dim / INT4_BLOCK));
            for i in 0..n {
                let codes = view.row_codes(i);
                for j in 0..dim {
                    let g = j / INT4_BLOCK;
                    let scale = view.group_scale(i, g);
                    assert!(scale.is_finite() && scale > 0.0);
                    let byte = codes[j / 2];
                    let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let code = nibble_to_i4(nibble);
                    assert!((-7..=7).contains(&code));
                    let x = rows[i * dim + j];
                    let y = code as f32 * scale;
                    // the scale is f16-rounded, so the block maximum may sit one
                    // f16 ulp past 7 * scale before clamping.
                    let tol = scale * 0.5 + x.abs() * (1.0 / 1024.0) + 1e-5;
                    assert!(
                        (x - y).abs() <= tol,
                        "row {i} col {j}: {x} vs {y} (scale {scale})"
                    );
                }
            }
        },
    );
}

#[test]
fn nibble_packing_is_exact() {
    property("nibble_packing_is_exact", |g| {
        let codes: Vec<i8> = (0..g.int(0, 511)).map(|_| g.int(0, 14) as i8 - 7).collect();
        let packed = pack_nibbles(&codes);
        assert_eq!(pack_nibbles(&codes), packed.clone());
        assert_eq!(packed.len(), codes.len().div_ceil(2));
        for (j, c) in codes.iter().enumerate() {
            let byte = packed[j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            assert_eq!(nibble_to_i4(nibble), *c);
        }
    });
}

// ---- wire dispatch: raw and intpack through `decode_payload` -------------

#[test]
fn raw_dispatch_is_identity_and_intpack_dispatch_never_panics() {
    property(
        "raw_dispatch_is_identity_and_intpack_dispatch_never_panics",
        |g| {
            let values = u64s(g);
            let raw: Vec<u8> = (0..g.int(0, 4095)).map(|_| g.u64() as u8).collect();
            assert_eq!(
                decode_payload(SECTION_ENCODING_RAW, &raw)
                    .expect("raw")
                    .into_owned(),
                raw.clone()
            );
            // the intpack wire codec is the chunk_ids / spans repack, which has
            // its own section-level tests; the primitive it is built on is the
            // u64 packer covered above. a bare packer payload is NOT a valid
            // repack container, so dispatch must reject it with a typed error,
            // never panic.
            let packed = pack_u64s(&values);
            let _ = decode_payload(SECTION_ENCODING_INTPACK, &packed);
        },
    );
}
