#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::reader::UrnaView;
use urna_format::writer::UrnaFileBuilder;

fn main() {
    let m = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: 4,
        n_chunks: 1,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let bytes = UrnaFileBuilder::new(m)
        .add_chunk(ChunkInput {
            canonical_text: "hi".into(),
            source_uri: "doc.txt".into(),
            byte_start: 0,
            byte_end: 2,
            embedding: vec![1.0, 0.0, 0.0, 0.0],
        })
        .build_bytes()
        .unwrap();

    let path = "crates/urna-format/tests/fixtures/golden_v1_minimal.urna";
    std::fs::write(path, &bytes).unwrap();

    let view = UrnaView::from_bytes(&bytes).unwrap();
    println!("GOLDEN_LEN = {}", bytes.len());
    println!("GOLDEN_FILE_HASH = {}", view.file_hash_hex());
    println!("GOLDEN_CONTENT_HASH = {}", view.content_hash_hex().unwrap());
    let ids = urna_format::sections::decode_chunk_ids(
        view.get_section_data(urna_format::layout::SECTION_CHUNK_IDS)
            .unwrap(),
        1,
    )
    .unwrap();
    println!("GOLDEN_CHUNK_ID = {}", ids[0]);
}
