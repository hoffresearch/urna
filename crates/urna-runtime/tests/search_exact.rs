#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_runtime::MmapUrnaFile;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

fn manifest(dim: u32, n: u64) -> Manifest {
    Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim,
        n_chunks: n,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    }
}

fn chunk(text: &str, uri: &str, start: u64, end: u64, emb: Vec<f32>) -> ChunkInput {
    ChunkInput {
        canonical_text: text.into(),
        source_uri: uri.into(),
        byte_start: start,
        byte_end: end,
        embedding: emb,
    }
}

fn build_axes_file(path: &PathBuf, dim: usize, n: usize) {
    let mut builder = UrnaFileBuilder::new(manifest(dim as u32, n as u64));
    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        builder = builder.add_chunk(chunk(
            &format!("text_{}", i),
            "doc.txt",
            (i * 10) as u64,
            ((i + 1) * 10) as u64,
            emb,
        ));
    }
    builder.write_to_path(path).unwrap();
}

#[test]
fn exact_search_returns_expected_chunk() {
    let path = tmp_path("rt_exact.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);

    let rt = MmapUrnaFile::open(&path).unwrap();
    assert_eq!(rt.embedding_dim(), 4);
    assert_eq!(rt.n_embeddings(), 3);

    let res = rt.search(&[1.0f32, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(res.recall, 1.0);
    assert_eq!(res.index_type, "exact");
    assert_eq!(res.k_requested, 1);
    assert_eq!(res.k_returned, 1);
    assert_eq!(res.hits.len(), 1);
    assert!(res.hits[0].chunk_id.starts_with("sha256:"));
    assert!((res.hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(res.hits[0].score_type, "cosine");
    assert_eq!(res.hits[0].index_type, "exact");
    assert!(!res.hits[0].reranked);
    assert!(res.hits[0].file_hash.starts_with("sha256:"));
    assert!(res.hits[0].content_hash.starts_with("sha256:"));
    assert!(
        res.hits[0]
            .citation_id
            .starts_with(&format!("urna://{}/", res.hits[0].content_hash)),
        "citation_id={}, content_hash={}",
        res.hits[0].citation_id,
        res.hits[0].content_hash
    );
    assert_eq!(res.hits[0].source_uri, "doc.txt");
    assert_eq!(res.hits[0].offset_start, 0);
    assert_eq!(res.hits[0].offset_end, 10);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn score_is_real_cosine() {
    let path = tmp_path("rt_score.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);

    let rt = MmapUrnaFile::open(&path).unwrap();
    // Query at 45° between axis 0 and axis 1: top hit ~0.7071 against either.
    let q: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0];
    let res = rt.search(&q, 3).unwrap();
    assert_eq!(res.hits.len(), 3);
    let top_score = res.hits[0].score;
    assert!(
        (top_score - (1.0 / 2f32.sqrt())).abs() < 1e-5,
        "top_score={}",
        top_score
    );
    // Score for axis 2 (orthogonal to query) must be ~0.
    let last_score = res.hits[2].score;
    assert!(last_score.abs() < 1e-5);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn truncated_when_k_lt_n() {
    let path = tmp_path("rt_trunc.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    let res = rt.search(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    assert!(res.truncated);
    assert_eq!(res.k_returned, 2);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn k_zero_fails() {
    let path = tmp_path("rt_k0.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.search(&[1.0, 0.0, 0.0, 0.0], 0).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn nan_query_fails() {
    let path = tmp_path("rt_nan.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.search(&[f32::NAN, 0.0, 0.0, 0.0], 1).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn inf_query_fails() {
    let path = tmp_path("rt_inf.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.search(&[f32::INFINITY, 0.0, 0.0, 0.0], 1).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn wrong_dim_query_fails() {
    let path = tmp_path("rt_dim.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    assert!(rt.search(&[1.0], 1).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn zero_vector_query_fails() {
    let path = tmp_path("rt_zero.urna");
    let _ = std::fs::remove_file(&path);
    build_axes_file(&path, 4, 3);
    let rt = MmapUrnaFile::open(&path).unwrap();
    let res = rt.search(&[0.0, 0.0, 0.0, 0.0], 1);
    assert!(matches!(
        res,
        Err(urna_runtime::RuntimeError::ZeroNormQuery)
    ));
    let _ = std::fs::remove_file(&path);
}
