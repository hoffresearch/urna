//! Negative paths for the int4 block-64 embeddings encoding (`encoding=7`):
//!
//! - Payload version other than 1 -> `UnsupportedSectionVersion`.
//! - Scale kind other than 1 (per-group) -> `MalformedSectionPayload`.
//! - Per-group f16 scale = NaN or Inf -> `InvalidEmbeddingValue` from
//!   `validate_embeddings_values`.
//! - Truncated prefix -> `EmbeddingSizeMismatch`.
//!
//! Mirrors `negative_int8.rs`. Layout reminder (see `encoding/int4.rs`):
//!
//! ```text
//!   u32 LE  payload_version = 1                       [0..4]
//!   u32 LE  scale_kind      = 1  (per-group block-64) [4..8]
//!   f16 LE  * (n * dim/64)                            [8..8+2*n*blocks]
//!   u8      * (n * dim/2)   packed nibbles            [...]
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::{
    SECTION_EMBEDDINGS, SECTION_ENCODING_INT4, URNA_FOOTER_SIZE, UrnaFooter,
};
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::writer::{EmbeddingDType, UrnaFileBuilder};
use urna_format::{ChunkInput, UrnaError, UrnaView};

const DIM: usize = 64; // one block per row.

fn manifest(n: u64, dim: u32) -> Manifest {
    Manifest {
        format_version: 1,
        schema_version: 1,
        embedding_model: "demo".into(),
        embedding_dim: dim,
        n_chunks: n,
        dtype: "float32".into(), // builder rewrites to "int4"
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
            v[(i + 1) % dim] = -0.5;
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

fn build_int4(n: usize, dim: usize) -> Vec<u8> {
    UrnaFileBuilder::new(manifest(n as u64, dim as u32))
        .embedding_dtype(EmbeddingDType::Int4)
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
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_unknown_payload_version() {
    let mut bytes = build_int4(2, DIM);
    let off = embeddings_offset(&bytes);
    bytes[off..off + 4].copy_from_slice(&99u32.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

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
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_unknown_scale_kind() {
    let mut bytes = build_int4(2, DIM);
    let off = embeddings_offset(&bytes);
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
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_nan_in_group_scale() {
    let mut bytes = build_int4(3, DIM);
    let off = embeddings_offset(&bytes);
    // First f16 group scale of row 0 sits right after the 8-byte prefix.
    // Set it to f16 NaN (0x7E00).
    bytes[off + 8..off + 10].copy_from_slice(&half::f16::NAN.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "expected InvalidEmbeddingValue for NaN group scale; got {:?}",
        res
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_inf_in_group_scale() {
    let mut bytes = build_int4(3, DIM);
    let off = embeddings_offset(&bytes);
    bytes[off + 8..off + 10].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
    rewrite_emb_checksum_and_file_hash(&mut bytes);

    let view = UrnaView::from_bytes(&bytes).unwrap();
    let res = view.validate_embeddings_values();
    assert!(
        matches!(res, Err(UrnaError::InvalidEmbeddingValue)),
        "expected InvalidEmbeddingValue for +Inf group scale; got {:?}",
        res
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn rejects_truncated_prefix() {
    // Truncate the embeddings section size in the table by one byte so the
    // payload no longer matches the int4 expected size. The reader's
    // validate_embeddings_layout surfaces EmbeddingSizeMismatch.
    let mut bytes = build_int4(2, DIM);
    let table_off = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[48..56].try_into().unwrap()) as usize;
    let entry_size = urna_format::layout::URNA_SECTION_ENTRY_SIZE;
    for i in 0..count {
        let off = table_off + i * entry_size;
        let sid = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        if sid == SECTION_EMBEDDINGS {
            let payload_off =
                u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
            let size = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
            let new_size = size - 1;
            bytes[off + 16..off + 24].copy_from_slice(&new_size.to_le_bytes());
            // recompute the section checksum over the truncated slice.
            use sha2::{Digest, Sha256};
            let h = Sha256::digest(&bytes[payload_off..payload_off + new_size as usize]);
            bytes[off + 24..off + 32].copy_from_slice(&h[..8]);
            break;
        }
    }
    let body_end = bytes.len() - URNA_FOOTER_SIZE;
    let new_file_hash = UrnaFooter::compute_file_hash(&bytes[..body_end]);
    bytes[body_end + 8..body_end + 40].copy_from_slice(&new_file_hash);

    let res = UrnaView::from_bytes(&bytes);
    assert!(
        matches!(res, Err(UrnaError::EmbeddingSizeMismatch { .. })),
        "expected EmbeddingSizeMismatch; got {:?}",
        res.err()
    );
}

#[test]
fn rejects_dim_not_multiple_of_block_at_section_level() {
    // Section-level (writer) rejection: an int4 build whose embedding_dim is
    // not a multiple of 64 fails to encode the embeddings payload, so no
    // file is produced (the view-level rejection is covered separately in
    // tests/int4_roundtrip.rs). dim = 63 -> not a multiple of INT4_BLOCK.
    let dim = 63usize;
    let res = UrnaFileBuilder::new(manifest(2, dim as u32))
        .embedding_dtype(EmbeddingDType::Int4)
        .reproducible(true)
        .add_chunks(unit_chunks(2, dim))
        .build_bytes();
    assert!(
        matches!(res, Err(UrnaError::InvalidInput(_))),
        "expected InvalidInput for dim not divisible by 64; got {:?}",
        res.err()
    );
}

#[test]
#[cfg_attr(all(miri, target_arch = "aarch64"), ignore)] // half converts f16 with inline asm on aarch64, which miri cannot run
fn int4_baseline_validates_and_has_correct_size() {
    let n = 4;
    let dim = 128; // 2 blocks per row.
    let bytes = build_int4(n, dim);
    let view = UrnaView::from_bytes(&bytes).unwrap();
    assert_eq!(view.manifest.dtype, "int4");
    let entry = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_EMBEDDINGS)
        .unwrap();
    assert_eq!(entry.encoding, SECTION_ENCODING_INT4);
    // 8-byte prefix + n * blocks f16 scales + n*dim/2 packed nibble bytes.
    let blocks = dim / 64;
    assert_eq!(entry.size as usize, 8 + n * blocks * 2 + n * dim / 2);
    view.validate_embeddings_values().unwrap();
}
