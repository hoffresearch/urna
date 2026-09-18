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
//! default 256 cases per property; `PROPTEST_CASES=5000` for a local soak.
//! failing inputs are persisted under `proptest-regressions/` and replayed
//! first on the next run.

// not under miri: proptest volume is not what miri is for: hundreds of cases per property, each a full encode + decode; the fixed-input codec tests cover the same paths under miri.
#![cfg(not(miri))]

use proptest::prelude::*;
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

// ---- strategies --------------------------------------------------------

/// integers across every intpack frame width: tiny, u32-range, full u64.
fn u64s() -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(
        prop_oneof![
            4 => 0u64..16,
            3 => 0u64..(u32::MAX as u64),
            1 => any::<u64>(),
        ],
        0..2000,
    )
}

/// canonical texts: empty, ascii, multibyte utf-8, long runs; 0..200 chunks.
fn texts() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        prop_oneof![
            1 => Just(String::new()),
            4 => "[a-z ,.]{1,120}",
            2 => "[\\p{L}\\p{N} ]{1,80}",
            1 => "(the quick brown fox |lorem ipsum dolor |urna://){1,6}[a-f0-9]{0,64}",
        ],
        0..200,
    )
}

/// texts with duplicates guaranteed: every entry is drawn from a pool of at
/// most 12 distinct strings.
fn texts_with_duplicates() -> impl Strategy<Value = Vec<String>> {
    (
        prop::collection::vec("[a-z ]{0,40}", 1..12),
        prop::collection::vec(0usize..12, 0..300),
    )
        .prop_map(|(pool, picks)| {
            picks
                .into_iter()
                .map(|i| pool[i % pool.len()].clone())
                .collect()
        })
}

/// enough boilerplate-rich material for `train_dict` to accept: the trainer
/// needs at least 8 unique samples and some shared substrings.
fn dict_corpus() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(
        "(patient presents with |dose adjusted to |follow-up in |see section ){1,4}[a-z0-9 ]{8,60}",
        16..80,
    )
}

/// `n` l2-normalized rows of `dim` finite f32 (no zero rows).
fn normalized_rows(
    dim: impl Strategy<Value = usize>,
) -> impl Strategy<Value = (Vec<f32>, usize, usize)> {
    (1usize..24, dim).prop_flat_map(|(n, dim)| {
        prop::collection::vec(-1.0f32..1.0, n * dim)
            .prop_filter("no zero row", move |v| {
                v.chunks_exact(dim).all(|r| r.iter().any(|x| *x != 0.0))
            })
            .prop_map(move |mut v| {
                for row in v.chunks_exact_mut(dim) {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    for x in row.iter_mut() {
                        *x /= norm;
                    }
                }
                (v, n, dim)
            })
    })
}

fn raw_canonical(texts: &[String]) -> Vec<u8> {
    encode_chunks_canonical(texts).expect("raw canonical encoding never fails")
}

// ---- intpack -----------------------------------------------------------

proptest! {
    #[test]
    fn intpack_roundtrips_and_is_byte_stable(values in u64s()) {
        let packed = pack_u64s(&values);
        prop_assert_eq!(pack_u64s(&values), packed.clone(), "encoder is not a pure function");
        prop_assert_eq!(unpack_u64s(&packed).expect("unpack"), values.clone());
        let reader = IntpackReader::parse(&packed).expect("parse");
        prop_assert_eq!(reader.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            prop_assert_eq!(reader.get(i).expect("in-range get"), *v);
        }
        prop_assert!(reader.get(values.len()).is_err(), "past-the-end get must be a typed error");
    }
}

// ---- txt_streams / fsst / zstd_dict: every text codec decodes to the raw
// ---- canonical payload, byte for byte -----------------------------------

proptest! {
    #[test]
    fn txt_streams_roundtrips_and_is_byte_stable(texts in texts()) {
        let raw = raw_canonical(&texts);
        let enc = encode_txt_streams(&texts).expect("encode");
        prop_assert_eq!(encode_txt_streams(&texts).expect("encode twice"), enc.clone());
        prop_assert_eq!(decode_txt_streams_payload(&enc).expect("decode"), raw.clone());
        prop_assert_eq!(
            decode_payload(SECTION_ENCODING_TXT_STREAMS, &enc).expect("dispatch").into_owned(),
            raw
        );
        let view = TxtStreams::parse(&enc).expect("parse");
        prop_assert_eq!(view.len(), texts.len());
        for (i, t) in texts.iter().enumerate() {
            prop_assert_eq!(&view.text(i).expect("text"), t);
        }
    }

    #[test]
    fn fsst_roundtrips_and_is_byte_stable(texts in texts()) {
        let raw = raw_canonical(&texts);
        let enc = encode_fsst(&texts).expect("encode");
        prop_assert_eq!(encode_fsst(&texts).expect("encode twice"), enc.clone());
        prop_assert_eq!(decode_fsst_payload(&enc).expect("decode"), raw.clone());
        prop_assert_eq!(
            decode_payload(SECTION_ENCODING_FSST, &enc).expect("dispatch").into_owned(),
            raw
        );
    }

    #[test]
    fn zstd_dict_roundtrips_and_is_byte_stable(texts in dict_corpus()) {
        let mut unique = texts.clone();
        unique.sort();
        unique.dedup();
        prop_assume!(unique.len() >= 8);
        let dict = train_dict(&unique);
        prop_assume!(dict.is_some());
        let dict = dict.expect("checked");
        prop_assert_eq!(train_dict(&unique).expect("train twice"), dict.clone(), "dictionary training is not deterministic");
        let raw = raw_canonical(&texts);
        let enc = encode_zstd_dict(&texts, &dict).expect("encode");
        prop_assert_eq!(encode_zstd_dict(&texts, &dict).expect("encode twice"), enc.clone());
        prop_assert_eq!(decode_zstd_dict_payload(&enc, &dict).expect("decode"), raw);
    }
}

// ---- dedup ---------------------------------------------------------------

proptest! {
    #[test]
    fn dedup_map_roundtrips_and_expands_exactly(texts in texts_with_duplicates()) {
        let d = dedup(&texts);
        let map = encode_dedup_map(&d.back_refs);
        prop_assert_eq!(encode_dedup_map(&d.back_refs), map.clone());
        let refs = decode_dedup_map(&map).expect("decode map");
        prop_assert_eq!(&refs, &d.back_refs);
        prop_assert_eq!(expand_dedup(&d.unique, &refs).expect("expand"), texts.clone());
        // the unique pool is first-seen order with no repeats.
        let mut seen = std::collections::HashSet::new();
        prop_assert!(d.unique.iter().all(|u| seen.insert(u.clone())));
        prop_assert!(d.unique.len() <= texts.len());
    }
}

// ---- embeddings: float16 / int8 / int4 -----------------------------------

proptest! {
    #[test]
    fn float16_roundtrips_within_one_ulp(values in prop::collection::vec(-1.0f32..1.0, 0..2048)) {
        let bytes = f32_to_f16_bytes(&values);
        prop_assert_eq!(f32_to_f16_bytes(&values), bytes.clone());
        prop_assert_eq!(bytes.len(), values.len() * 2);
        let back = f16_bytes_to_f32(&bytes);
        for (x, y) in values.iter().zip(back.iter()) {
            // f16 has 11 significand bits: relative error <= 2^-11 (+ a
            // subnormal floor near zero).
            let tol = x.abs() * (1.0 / 2048.0) + 1e-6;
            prop_assert!((x - y).abs() <= tol, "{x} -> {y} exceeds f16 tolerance {tol}");
        }
    }

    #[test]
    fn int8_payload_is_sized_stable_and_reconstructs_within_half_a_step(
        (rows, n, dim) in normalized_rows(1usize..512)
    ) {
        let enc = encode_int8_embeddings(&rows, n, dim).expect("encode");
        prop_assert_eq!(encode_int8_embeddings(&rows, n, dim).expect("encode twice"), enc.clone());
        prop_assert_eq!(Some(enc.len()), expected_embeddings_size("int8", n, dim));
        prop_assert_eq!(
            decode_payload(SECTION_ENCODING_INT8, &enc).expect("dispatch").len(),
            enc.len(),
            "embedding encodings are their own canonical bytes"
        );
        let view = Int8EmbeddingsView::parse(&enc, n, dim).expect("parse");
        prop_assert_eq!((view.n, view.dim), (n, dim));
        for i in 0..n {
            let scale = view.scale(i);
            prop_assert!(scale.is_finite() && scale > 0.0);
            let row = view.row(i);
            for (j, code) in row.iter().enumerate() {
                let x = rows[i * dim + j];
                let y = *code as f32 * scale;
                prop_assert!((x - y).abs() <= scale * 0.5 + 1e-5, "row {i} col {j}: {x} vs {y} (scale {scale})");
            }
        }
    }

    #[test]
    fn int4_payload_is_sized_stable_and_reconstructs_within_half_a_step(
        (rows, n, dim) in normalized_rows((1usize..=8).prop_map(|k| k * INT4_BLOCK))
    ) {
        let enc = encode_int4_embeddings(&rows, n, dim).expect("encode");
        prop_assert_eq!(encode_int4_embeddings(&rows, n, dim).expect("encode twice"), enc.clone());
        prop_assert_eq!(Some(enc.len()), expected_embeddings_size("int4", n, dim));
        prop_assert_eq!(
            decode_payload(SECTION_ENCODING_INT4, &enc).expect("dispatch").len(),
            enc.len()
        );
        let view = Int4EmbeddingsView::parse(&enc, n, dim).expect("parse");
        prop_assert_eq!((view.n, view.dim, view.blocks), (n, dim, dim / INT4_BLOCK));
        for i in 0..n {
            let codes = view.row_codes(i);
            for j in 0..dim {
                let g = j / INT4_BLOCK;
                let scale = view.group_scale(i, g);
                prop_assert!(scale.is_finite() && scale > 0.0);
                let byte = codes[j / 2];
                let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let code = nibble_to_i4(nibble);
                prop_assert!((-7..=7).contains(&code));
                let x = rows[i * dim + j];
                let y = code as f32 * scale;
                // the scale is f16-rounded, so the block maximum may sit one
                // f16 ulp past 7 * scale before clamping.
                let tol = scale * 0.5 + x.abs() * (1.0 / 1024.0) + 1e-5;
                prop_assert!((x - y).abs() <= tol, "row {i} col {j}: {x} vs {y} (scale {scale})");
            }
        }
    }

    #[test]
    fn nibble_packing_is_exact(codes in prop::collection::vec(-7i8..=7, 0..512)) {
        let packed = pack_nibbles(&codes);
        prop_assert_eq!(pack_nibbles(&codes), packed.clone());
        prop_assert_eq!(packed.len(), codes.len().div_ceil(2));
        for (j, c) in codes.iter().enumerate() {
            let byte = packed[j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            prop_assert_eq!(nibble_to_i4(nibble), *c);
        }
    }
}

// ---- wire dispatch: raw and intpack through `decode_payload` -------------

proptest! {
    #[test]
    fn raw_dispatch_is_identity_and_intpack_dispatch_never_panics(values in u64s(), raw in prop::collection::vec(any::<u8>(), 0..4096)) {
        prop_assert_eq!(decode_payload(SECTION_ENCODING_RAW, &raw).expect("raw").into_owned(), raw.clone());
        // the intpack wire codec is the chunk_ids / spans repack, which has
        // its own section-level tests; the primitive it is built on is the
        // u64 packer covered above. a bare packer payload is NOT a valid
        // repack container, so dispatch must reject it with a typed error,
        // never panic.
        let packed = pack_u64s(&values);
        let _ = decode_payload(SECTION_ENCODING_INTPACK, &packed);
    }
}
