//! Golden file test: a frozen `.urna` file that any future change must
//! either preserve byte-for-byte or explicitly regenerate.
//!
//! The fixture at `tests/fixtures/golden_v1_minimal.urna` is built by the
//! same builder used by the rest of the suite, with these inputs:
//!
//! ```text
//!   manifest:
//!     format_version: 1
//!     schema_version: 1
//!     embedding_model: "demo"
//!     embedding_dim: 4
//!     n_chunks: 1
//!     dtype: "float32"
//!     metric: "ip"
//!     score_type: "cosine"
//!     normalize: "l2"
//!     index_type: "exact"
//!     rerank_policy: "none"
//!     model_hash: sha256:00...00 (64 zeros)
//!     chunker_version: "demo-chunker/1"
//!     capabilities: defaults (exact + reproducible)
//!
//!   chunk[0]:
//!     canonical_text: "hi"
//!     source_uri: "doc.txt"
//!     byte_start: 0
//!     byte_end: 2
//!     embedding: [1.0, 0.0, 0.0, 0.0]
//! ```
//!
//! Layout offsets (informational; checked transitively by the structural
//! assertions below):
//!
//! ```text
//!   [0   .. 4)   magic "URNA"
//!   [4   .. 6)   version_major 0x0001
//!   [6   .. 8)   version_minor 0x0000
//!   [128 .. 320) section table (6 entries × 32 bytes)
//!   [320 .. ..)  manifest JSON (canonical, no whitespace)
//!   [..  .. ..)  six required sections, in section_id order
//!   [last-40)    footer (u64 footer_size + 32-byte file_hash)
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::layout::*;
use urna_format::sections::{
    decode_chunk_ids, decode_chunks_canonical, decode_chunks_original_spans, decode_search_contract,
};
use urna_format::{ChunkInput, UrnaFileBuilder, UrnaView};

const GOLDEN: &[u8] = include_bytes!("fixtures/golden_v1_minimal.urna");

const GOLDEN_FILE_HASH: &str =
    "sha256:2c8dad39b1b9bc440d6a8d48f9b5e6d071bc2c66076be719c20bc6de1d502390";
const GOLDEN_CONTENT_HASH: &str =
    "sha256:982e986094ae9c3f14ccdb19faf0d27d8f9155bfc53b025b37b0624fab78a378";
const GOLDEN_CHUNK_ID: &str =
    "sha256:186d842224fbbee05b4ae5d2851ef966c7da8a54372517c450c2a5d35dd213c3";
const GOLDEN_LEN: usize = 1366;

#[test]
fn golden_file_size_unchanged() {
    assert_eq!(
        GOLDEN.len(),
        GOLDEN_LEN,
        "golden file size changed; if intentional, regenerate the fixture",
    );
}

#[test]
fn golden_file_header_bytes() {
    assert_eq!(&GOLDEN[0..4], b"URNA");
    assert_eq!(u16::from_le_bytes([GOLDEN[4], GOLDEN[5]]), 1); // version_major
    assert_eq!(u16::from_le_bytes([GOLDEN[6], GOLDEN[7]]), 0); // version_minor
}

#[test]
fn golden_file_hashes_unchanged() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    assert_eq!(view.file_hash_hex(), GOLDEN_FILE_HASH);
    assert_eq!(view.content_hash_hex().unwrap(), GOLDEN_CONTENT_HASH);
}

#[test]
fn golden_file_parses_with_full_validation() {
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    assert_eq!(view.header.embedding_dim, 4);
    assert_eq!(view.header.n_chunks, 1);
    assert_eq!(view.header.n_embeddings, 1);
    assert_eq!(view.section_table.len(), 6);
    // Every section payload starts on a SECTION_ALIGNMENT (64B) boundary.
    for entry in &view.section_table {
        assert_eq!(
            entry.offset % SECTION_ALIGNMENT,
            0,
            "section {} offset {} not aligned to {}",
            entry.section_id,
            entry.offset,
            SECTION_ALIGNMENT
        );
        assert_eq!(
            entry.encoding, SECTION_ENCODING_RAW,
            "section {} has non-raw encoding {}",
            entry.section_id, entry.encoding
        );
    }
    assert_eq!(view.manifest.embedding_model, "demo");
    assert_eq!(view.manifest.dtype, "float32");
    assert_eq!(view.manifest.metric, "ip");
    assert_eq!(view.manifest.score_type, "cosine");
    assert_eq!(view.manifest.normalize, "l2");
    assert_eq!(view.manifest.index_type, "exact");
    assert_eq!(view.manifest.rerank_policy, "none");
    assert!(view.manifest.capabilities.supports_exact);
    assert!(view.manifest.capabilities.supports_reproducible_build);

    let ids = decode_chunk_ids(view.get_section_data(SECTION_CHUNK_IDS).unwrap(), 1).unwrap();
    assert_eq!(ids, vec![GOLDEN_CHUNK_ID.to_string()]);

    let texts =
        decode_chunks_canonical(view.get_section_data(SECTION_CHUNKS_CANONICAL).unwrap(), 1)
            .unwrap();
    assert_eq!(texts, vec!["hi".to_string()]);

    let spans = decode_chunks_original_spans(
        view.get_section_data(SECTION_CHUNKS_ORIGINAL_SPANS)
            .unwrap(),
        1,
    )
    .unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].source_uri, "doc.txt");
    assert_eq!(spans[0].byte_start, 0);
    assert_eq!(spans[0].byte_end, 2);

    let contract =
        decode_search_contract(view.get_section_data(SECTION_SEARCH_CONTRACT).unwrap()).unwrap();
    assert_eq!(contract.metric, "ip");
    assert_eq!(contract.score_type, "cosine");
    assert_eq!(contract.normalize, "l2");
    assert_eq!(contract.index_type, "exact");
    assert_eq!(contract.rerank_policy, "none");
}

#[test]
fn golden_chunks_canonical_stays_raw_not_txt_streams() {
    // the txt_streams codec (encoding id 10) must NEVER touch raw-text
    // presets. the golden uses the default (raw) text encoding, so its
    // chunks_canonical section stays SECTION_ENCODING_RAW and the file is
    // byte-identical at 1366 bytes. this is the content_hash + citation
    // stability guard for the per-chunk-streams re-layout.
    assert_eq!(GOLDEN.len(), GOLDEN_LEN);
    let view = UrnaView::from_bytes(GOLDEN).unwrap();
    let enc = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_CHUNKS_CANONICAL)
        .unwrap()
        .encoding;
    assert_eq!(
        enc, SECTION_ENCODING_RAW,
        "golden chunks_canonical must stay raw, never txt_streams"
    );
}

#[test]
fn rebuilding_reproduces_golden() {
    use urna_format::manifest::Manifest;
    let m = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let rebuilt = UrnaFileBuilder::new(m)
        .add_chunk(ChunkInput {
            canonical_text: "hi".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 2,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .build_bytes()
        .unwrap();
    assert_eq!(
        rebuilt.len(),
        GOLDEN.len(),
        "rebuilt size differs from golden"
    );
    assert_eq!(rebuilt, GOLDEN, "rebuilt bytes diverge from golden");
}
