//! Negative paths for the int8 embeddings encoding (`encoding=3`):
//!
//! - Payload version other than 1 → `UnsupportedSectionVersion`.
//! - Scale kind other than 0 (per-vector f32) → `MalformedSectionPayload`.
//! - Per-vector scale = NaN or Inf → `InvalidEmbeddingValue` from
//!   `validate_embeddings_values`.
//! - Truncated prefix (less than `INT8_PREFIX_SIZE` bytes) →
//!   `EmbeddingSizeMismatch`.
//!
//! Layout reminder (see `encoding/int8.rs`):
//!
//! ```text
//!   u32 LE  payload_version = 1                    [0..4]
//!   u32 LE  scale_kind      = 0  (per-vector f32)  [4..8]
//!   f32 LE  * n                                     [8..8+4n]
//!   i8      * (n * dim)                             [8+4n..]
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{
    SECTION_EMBEDDINGS, SECTION_ENCODING_INT8, URNA_FOOTER_SIZE, UrnaFooter,
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
        dtype: "float32".into(), // builder rewrites to "int8"
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

fn build_int8(n: usize, dim: usize) -> Vec<u8> {
    UrnaFileBuilder::new(manifest(n as u64, dim as u32))
        .embedding_dtype(EmbeddingDType::Int8)
        .reproducible(true)
        .add_chunks(unit_chunks(n, dim))
        .build_bytes()
        .unwrap()
}

fn embeddings_offset(bytes: &[u8]) -> usize {
    let table_off = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
    let entry_size = urna_format::layout::URNA_SECTION_ENTRY_SIZE;
    for i in 0..count {
        let off = table_off + i * entry_size;
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if sid == SECTION_EMBEDDINGS {
            return u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
        }
    }
    panic!("embeddings section not found");
}

/// Recompute the embeddings section's checksum and the file_hash. See
/// negative_fp16/zstd for the same pattern; centralized here too to
/// keep the test self-contained.
fn rewrite_emb_checksum_and_file_hash(bytes: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let table_off = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
    let entry_size = urna_format::layout::URNA_SECTION_ENTRY_SIZE;
    for i in 0..count {
        let off = table_off + i * entry_size;
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if sid == SECTION_EMBEDDINGS {
            let payload_off =
                u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
            let payload_size =
                u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap()) as usize;
            let h = Sha256::digest(&bytes[payload_off..payload_off + payload_size]);
            bytes[off + 24..off + 32].copy_from_slice(&h[..8]);
            break;
        }
    }
    let body_end = bytes.len() - URNA_FOOTER_SIZE;
    let new_file_hash = UrnaFooter::compute_file_hash(&bytes[..body_end]);
    bytes[body_end + 8..body_end + 40].copy_from_slice(&new_file_hash);
}

#[test]
fn rejects_unknown_payload_version() {
    let mut bytes = build_int8(2, 4);
    let off = embeddings_offset(&bytes);
    // Set payload_version = 99 (unknown).
    bytes[off..off + 4].copy_from_slice(&99u32.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    // Reader's validate_embeddings_layout / values is what surfaces this
    // since the int8 prefix is parsed lazily.
    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(
            res,
            Err(UrnaError::UnsupportedSectionVersion {
                section_id: SECTION_EMBEDDINGS,
                version: 99
            })
        ),
        "expected UnsupportedSectionVersion(99); got {:?}",
        res
    );
}

#[test]
fn rejects_unknown_scale_kind() {
    let mut bytes = build_int8(2, 4);
    let off = embeddings_offset(&bytes);
    // Set scale_kind = 99 (no quantizer defined).
    bytes[off + 4..off + 8].copy_from_slice(&99u32.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(
            res,
            Err(UrnaError::MalformedSectionPayload {
                section_id: SECTION_EMBEDDINGS,
                ..
            })
        ),
        "expected MalformedSectionPayload(scale_kind 99); got {:?}",
        res
    );
}

#[test]
fn rejects_nan_in_per_vector_scale() {
    let mut bytes = build_int8(3, 4);
    let off = embeddings_offset(&bytes);
    // Scale[1] (second vector) = NaN. Scales start at offset 8 of the
    // payload; each is 4 bytes.
    bytes[off + 8 + 4..off + 8 + 8].copy_from_slice(&f32::NAN.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "expected InvalidEmbeddingValue for NaN scale; got {:?}",
        res
    );
}

#[test]
fn rejects_inf_in_per_vector_scale() {
    let mut bytes = build_int8(3, 4);
    let off = embeddings_offset(&bytes);
    bytes[off + 8..off + 8 + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "expected InvalidEmbeddingValue for +Inf scale; got {:?}",
        res
    );
}

#[test]
fn int8_baseline_validates_and_has_correct_size() {
    let n = 4;
    let dim = 8;
    let bytes = build_int8(n, dim);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(view.manifest.dtype, "int8");
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_INT8);
    // 8-byte prefix + n f32 scales + n*dim i8 bodies.
    assert_eq!(entry.size as usize, 8 + n * 4 + n * dim);
    view.validate_embeddings_values().unwrap();
}

#[test]
fn int8_odd_dim_validates_cleanly() {
    for &dim in &[5usize, 7, 11, 13] {
        let bytes = build_int8(6, dim);
        let view = UrnaView::from_bytes(&bytes).unwrap();
        view.validate_embeddings_values()
            .unwrap_or_else(|e| panic!("int8 dim={} failed: {:?}", dim, e));
    }
}
