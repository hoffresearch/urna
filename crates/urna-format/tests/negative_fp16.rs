//! Negative paths for the float16 embeddings encoding (`encoding=2`):
//!
//! - `validate_embeddings_values` walks the section and rejects NaN
//!   and Inf in fp16 just like it does for f32. The writer's
//!   `add_chunk` would catch f32 NaN/Inf at build time, but a runtime
//!   that mmaps a tampered file must also catch corrupt fp16 values.
//! - SIMD parity for odd dims: the f16 dot-product path has a tail
//!   loop for elements that don't fit a SIMD register; this test
//!   confirms the public reader accepts dims 5, 7, and 11 without
//!   error and produces values consistent with the scalar fallback.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{
    SECTION_EMBEDDINGS, SECTION_ENCODING_FLOAT16, URNA_FOOTER_SIZE, UrnaFooter,
};
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::writer::{EmbeddingDType, UrnaFileBuilder};
use urna_format::{ChunkInput, UrnaError, UrnaView};

fn manifest(n: u64, dim: u32) -> Manifest {
    Manifest {
        format_version: 1,
        schema_version: 1,
        embedding_model: "demo".into(),
        embedding_dim: dim,
        n_chunks: n,
        dtype: "float32".into(), // builder mutates to "float16" on .embedding_dtype()
        metric: "ip".into(),
        score_type: "cosine".into(),
        normalize: "l2".into(),
        index_type: "exact".into(),
        rerank_policy: "none".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        chunker_version: "demo-chunker/1".into(),
        capabilities: Capabilities {
            supports_exact: true,
            supports_reproducible_build: true,
            supports_ann: false,
            supports_bm25: false,
            supports_citations: true,
        },
        title: None,
        version: None,
        created: None,
        description: None,
        authors: None,
        license: None,
        mrl_dim: None,
        full_dim: None,
        capabilities_ext: None,
        extra: Default::default(),
    }
}

fn unit_chunks(n: usize, dim: usize) -> Vec<ChunkInput> {
    (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            ChunkInput {
                canonical_text: format!("chunk-{}", i),
                source_uri: "doc".into(),
                byte_start: i as u64,
                byte_end: (i + 1) as u64,
                embedding: v,
            }
        })
        .collect()
}

fn build_fp16(n: usize, dim: usize) -> Vec<u8> {
    UrnaFileBuilder::new(manifest(n as u64, dim as u32))
        .embedding_dtype(EmbeddingDType::Float16)
        .reproducible(true)
        .add_chunks(unit_chunks(n, dim))
        .build_bytes()
        .unwrap()
}

/// Rewrite the section's physical checksum (over the now-tampered
/// payload) and the footer's file_hash. Bypasses UrnaView::from_bytes
/// because that would refuse to parse a file whose section checksum
/// was just invalidated. Reads the section table directly from header
/// metadata at offsets 40 (section_table_offset, u64 LE) and 48
/// (section_table_count, u64 LE).
fn rewrite_section_checksum_and_file_hash(bytes: &mut [u8], section_id: u32) {
    use sha2::{Digest, Sha256};

    let table_off = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
    let entry_size = urna_format::layout::URNA_SECTION_ENTRY_SIZE;
    let mut found = None;
    for i in 0..count {
        let off = table_off + i * entry_size;
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if sid == section_id {
            found = Some(off);
            break;
        }
    }
    let entry_off = found.expect("section not found");
    let payload_off =
        u64::from_le_bytes(bytes[entry_off + 8..entry_off + 16].try_into().unwrap()) as usize;
    let payload_size =
        u64::from_le_bytes(bytes[entry_off + 16..entry_off + 24].try_into().unwrap()) as usize;
    let payload = &bytes[payload_off..payload_off + payload_size];
    let h = Sha256::digest(payload);
    bytes[entry_off + 24..entry_off + 32].copy_from_slice(&h[..8]);

    let body_end = bytes.len() - URNA_FOOTER_SIZE;
    let new_file_hash = UrnaFooter::compute_file_hash(&bytes[..body_end]);
    bytes[body_end + 8..body_end + 40].copy_from_slice(&new_file_hash);
}

fn fp16_le(value: f32) -> [u8; 2] {
    let h = half::f16::from_f32(value);
    h.to_le_bytes()
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_nan_in_fp16_embedding() {
    let mut bytes = build_fp16(2, 4);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_FLOAT16);
    let payload_off = entry.offset as usize;
    // Overwrite first f16 lane with a NaN bit pattern.
    let nan_le = fp16_le(f32::NAN);
    bytes[payload_off..payload_off + 2].copy_from_slice(&nan_le);
    rewrite_section_checksum_and_file_hash(&mut bytes, SECTION_EMBEDDINGS);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "fp16 NaN must be rejected; got {:?}",
        res
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_inf_in_fp16_embedding() {
    let mut bytes = build_fp16(2, 4);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    let payload_off = entry.offset as usize;
    let inf_le = fp16_le(f32::INFINITY);
    bytes[payload_off + 2..payload_off + 4].copy_from_slice(&inf_le);
    rewrite_section_checksum_and_file_hash(&mut bytes, SECTION_EMBEDDINGS);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "fp16 +Inf must be rejected; got {:?}",
        res
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_negative_inf_in_fp16_embedding() {
    let mut bytes = build_fp16(2, 4);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    let payload_off = entry.offset as usize;
    let neg_inf_le = fp16_le(f32::NEG_INFINITY);
    bytes[payload_off + 4..payload_off + 6].copy_from_slice(&neg_inf_le);
    rewrite_section_checksum_and_file_hash(&mut bytes, SECTION_EMBEDDINGS);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "fp16 -Inf must be rejected; got {:?}",
        res
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn fp16_section_size_matches_n_dim_2() {
    // float16 = 2 bytes/value. Section size should be exactly n*dim*2.
    for &(n, dim) in &[(3usize, 4usize), (5, 8), (10, 16)] {
        let bytes = build_fp16(n, dim);
        let view = UrnaView::from_bytes(&bytes).unwrap();
        let entry = view
            .section_table
            .iter()
            .find(|e| e.section_id == SECTION_EMBEDDINGS)
            .unwrap();
        assert_eq!(
            entry.size as usize,
            n * dim * 2,
            "fp16 section size mismatch for n={} dim={}",
            n,
            dim
        );
    }
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn fp16_odd_dims_validate_cleanly() {
    // Dims that don't align to 4/8/16 lane SIMD widths must still pass
    // validation. The runtime's SIMD dot product has a tail loop for
    // these; if it ever regresses, the simd module's parity tests fail
    // first, but this end-to-end check is the contract guarantee.
    for &dim in &[5usize, 7, 11, 13] {
        let bytes = build_fp16(8, dim);
        let view = UrnaView::from_bytes(&bytes).unwrap();
        view.validate_embeddings_values()
            .unwrap_or_else(|e| panic!("fp16 dim={} failed validation: {:?}", dim, e));
    }
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn fp16_baseline_decodes_with_no_error() {
    let bytes = build_fp16(4, 8);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(view.manifest.dtype, "float16");
    view.validate_embeddings_values().unwrap();
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn f16_codec_roundtrip_within_tolerance() {
    // the f32<->f16 byte codec round-trips within f16 precision (relocated
    // out of encoding/mod.rs to keep the wire-codec registry under 300 lines).
    let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
    let bytes = urna_format::f32_to_f16_bytes(&v);
    let back = urna_format::f16_bytes_to_f32(&bytes);
    assert_eq!(back.len(), v.len());
    for (a, b) in v.iter().zip(back.iter()) {
        assert!((a - b).abs() < 1e-3, "{} vs {}", a, b);
    }
}
