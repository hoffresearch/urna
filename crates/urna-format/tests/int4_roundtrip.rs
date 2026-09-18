//! int4 block-64 codec round-trip and unit coverage (`encoding=7`).
//!
//! These tests live here (not inline in `encoding/int4.rs`) so the codec
//! source stays under the 300-line rust src guard while keeping full
//! coverage of pack/unpack, quantize clamping, the section view, and the
//! typed malformed-payload rejections. They drive the public api exactly
//! as a downstream crate would.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{SECTION_EMBEDDINGS, SECTION_ENCODING_INT4};
use urna_format::{
    INT4_BLOCK, Int4EmbeddingsView, UrnaError, decode_payload, encode_int4_embeddings,
    expected_embeddings_size, int4_blocks_per_row, nibble_to_i4, pack_nibbles, quantize_f32_to_i4,
};

const INT4_PAYLOAD_VERSION: u32 = 1;
const INT4_SCALE_KIND_PER_GROUP: u32 = 1;

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn nibble_pack_unpack_exact_and_quantize_clamps_symmetric() {
    // pack/unpack is exact across the full 4-bit signed range.
    let all: Vec<i8> = (-8..=7).collect(); // 16 codes -> 8 bytes, even.
    let packed = pack_nibbles(&all);
    for (k, &orig) in all.iter().enumerate() {
        let nib = if k % 2 == 0 {
            packed[k / 2]
        } else {
            packed[k / 2] >> 4
        };
        assert_eq!(nibble_to_i4(nib), orig, "code {orig} round-trip");
    }
    // quantize clamps to the symmetric [-7, 7]; -8 is never emitted.
    let mut v = vec![0.0f32; INT4_BLOCK];
    (v[0], v[1], v[2]) = (1.0, -1.0, 0.5);
    let (scales, codes) = quantize_f32_to_i4(&v, INT4_BLOCK);
    assert_eq!((scales.len(), codes.len()), (1, INT4_BLOCK));
    assert!(codes.iter().all(|&c| (-7..=7).contains(&c)));
    assert!(codes.iter().any(|&c| c == 7 || c == -7));
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn quantize_round_trips_within_group_scale_tolerance() {
    // dim = 128 -> 2 blocks. Each component reconstructs within one
    // half-step of its own block's f16 scale.
    let dim = 128;
    let raw: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.013).sin()).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    let unit: Vec<f32> = raw.iter().map(|x| x / norm).collect();
    let (scales, codes) = quantize_f32_to_i4(&unit, dim);
    for (j, (&orig, &c)) in unit.iter().zip(codes.iter()).enumerate() {
        let scale = scales[j / INT4_BLOCK].to_f32();
        assert!(
            (orig - c as f32 * scale).abs() <= scale * 0.51 + 1e-6,
            "dim {j}"
        );
    }
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn section_roundtrip_view_matches_quantization() {
    let (n, dim) = (3usize, 128usize); // 2 blocks per row.
    let mut emb: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        (v[i], v[dim - 1 - i]) = (1.0, -0.5);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        emb.extend_from_slice(&v);
    }
    let payload = encode_int4_embeddings(&emb, n, dim).unwrap();
    let view = Int4EmbeddingsView::parse(&payload, n, dim).unwrap();
    assert_eq!(
        (view.n, view.dim, view.blocks),
        (n, dim, int4_blocks_per_row(dim))
    );
    for i in 0..n {
        let (scales, codes) = quantize_f32_to_i4(&emb[i * dim..(i + 1) * dim], dim);
        for (g, s) in scales.iter().enumerate() {
            assert_eq!(view.group_scale(i, g), s.to_f32());
        }
        let row = view.row_codes(i);
        for (j, &c) in codes.iter().enumerate() {
            let nib = if j % 2 == 0 {
                row[j / 2]
            } else {
                row[j / 2] >> 4
            };
            assert_eq!(nibble_to_i4(nib), c, "row {i} code {j}");
        }
    }
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn parse_rejects_malformed_payloads_with_typed_errors() {
    let n = 2;
    let dim = 64;
    let payload = encode_int4_embeddings(&vec![0.0; n * dim], n, dim).unwrap();
    // truncated and oversized both mismatch the expected size.
    assert!(matches!(
        Int4EmbeddingsView::parse(&payload[..payload.len() - 1], n, dim),
        Err(UrnaError::EmbeddingSizeMismatch { .. })
    ));
    let mut over = payload.clone();
    over.push(0);
    assert!(matches!(
        Int4EmbeddingsView::parse(&over, n, dim),
        Err(UrnaError::EmbeddingSizeMismatch { .. })
    ));
    // bad scale_kind, then bad payload_version.
    let mut bad = payload.clone();
    bad[4..8].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(
        Int4EmbeddingsView::parse(&bad, n, dim),
        Err(UrnaError::MalformedSectionPayload { .. })
    ));
    bad[4..8].copy_from_slice(&INT4_SCALE_KIND_PER_GROUP.to_le_bytes());
    bad[0..4].copy_from_slice(&7u32.to_le_bytes());
    assert!(matches!(
        Int4EmbeddingsView::parse(&bad, n, dim),
        Err(UrnaError::UnsupportedSectionVersion { .. })
    ));
    // dim not a multiple of the block size is rejected before sizing.
    assert!(matches!(
        Int4EmbeddingsView::parse(&[0u8; 64], 1, 63),
        Err(UrnaError::MalformedSectionPayload {
            section_id: SECTION_EMBEDDINGS,
            ..
        })
    ));
    // sanity: the prefix the encoder writes carries the expected version.
    assert_eq!(
        u32::from_le_bytes(payload[0..4].try_into().unwrap()),
        INT4_PAYLOAD_VERSION
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn int4_decode_payload_borrows_bytes() {
    // int4 (id 7) IS its own canonical bytes (like int8); decode_payload
    // must borrow, not copy, so the runtime scores it straight off mmap.
    let n = 2;
    let dim = 64;
    let payload = encode_int4_embeddings(&vec![0.0f32; n * dim], n, dim).unwrap();
    let decoded = decode_payload(SECTION_ENCODING_INT4, &payload).unwrap();
    assert!(matches!(decoded, std::borrow::Cow::Borrowed(_)));
    assert_eq!(decoded.as_ref(), payload.as_slice());
    assert_eq!(
        expected_embeddings_size("int4", n, dim),
        Some(payload.len())
    );
}
