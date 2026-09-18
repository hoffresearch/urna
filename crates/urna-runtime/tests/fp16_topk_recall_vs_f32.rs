//! Top-k recall regression: float16 embeddings must approximate the
//! float32 ranking closely enough to be production-safe.
//!
//! Why not require an exact ID match? Quantizing f32 → f16 introduces
//! a small per-component rounding (≤ 5e-4 relative). Two near-tied
//! results in the f32 ranking may swap positions in f16 within that
//! tolerance. Demanding "top-k matches exactly" turns into a flaky
//! test on real data; demanding `recall@10 ≥ 0.98` (≥ 9.8 IDs per 10
//! shared on average) is the honest contract.
//!
//! We also bound the score drift between corresponding hits to a
//! conservative 1e-3, since fp16 round-trip on L2-normalized vectors
//! tends to land in the 1e-4..1e-5 range. If drift ever exceeds 1e-3
//! the SIMD f32-accumulation invariant has regressed.
//!
//! Determinism: `Lcg` from `urna_runtime::ann` is reused so the
//! corpus is identical to the one in `hnsw_recall.rs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use urna_format::ChunkInput;
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::writer::{EmbeddingDType, UrnaFileBuilder};
use urna_runtime::MmapUrnaFile;

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(
            seed.wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        )
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))) as f32
    }
}

fn random_l2(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(rng.next_f32() - 0.5);
        }
        let norm: f32 = v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for x in &mut v {
            *x /= norm;
        }
        out.push(v);
    }
    out
}

fn manifest(n: u64, dim: u32) -> Manifest {
    Manifest {
        format_version: 1,
        schema_version: 1,
        embedding_model: "demo".into(),
        embedding_dim: dim,
        n_chunks: n,
        dtype: "float32".into(),
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

fn corpus_chunks(vectors: &[Vec<f32>]) -> Vec<ChunkInput> {
    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| ChunkInput {
            canonical_text: format!("chunk-{}", i),
            source_uri: "synthetic".into(),
            byte_start: i as u64,
            byte_end: (i + 1) as u64,
            embedding: v.clone(),
        })
        .collect()
}

fn write_urna(dtype: EmbeddingDType, vectors: &[Vec<f32>], dim: u32) -> PathBuf {
    let bytes = UrnaFileBuilder::new(manifest(vectors.len() as u64, dim))
        .embedding_dtype(dtype)
        .reproducible(true)
        .add_chunks(corpus_chunks(vectors))
        .build_bytes()
        .unwrap();
    let id = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "urna_fp16_recall_{}_{}.urna",
        std::process::id(),
        id
    ));
    std::fs::write(&path, &bytes).unwrap();
    path
}

#[test]
fn fp16_recall_at_10_geq_098_vs_f32() {
    let n = 1000;
    let dim = 64;
    let n_queries = 50;
    let k: i32 = 10;

    let corpus = random_l2(n, dim, 0xCAFE_BABE);
    let queries = random_l2(n_queries, dim, 0xFACE_FEED);

    let f32_path = write_urna(EmbeddingDType::Float32, &corpus, dim as u32);
    let f16_path = write_urna(EmbeddingDType::Float16, &corpus, dim as u32);

    let f32_db = MmapUrnaFile::open(&f32_path).unwrap();
    let f16_db = MmapUrnaFile::open(&f16_path).unwrap();

    let mut total_recall = 0.0f64;
    let mut max_drift = 0.0f32;
    let mut sum_drift = 0.0f64;
    let mut drift_samples = 0usize;

    for q in &queries {
        let r32 = f32_db.search(q, k).unwrap();
        let r16 = f16_db.search(q, k).unwrap();
        let ids32: HashSet<&str> = r32.hits.iter().map(|h| h.chunk_id.as_str()).collect();
        let ids16: HashSet<&str> = r16.hits.iter().map(|h| h.chunk_id.as_str()).collect();
        let overlap = ids32.intersection(&ids16).count();
        total_recall += overlap as f64 / k as f64;

        // Score drift: same chunk_id should score within 1e-3 across dtypes.
        let by_id_32: std::collections::HashMap<&str, f32> = r32
            .hits
            .iter()
            .map(|h| (h.chunk_id.as_str(), h.score))
            .collect();
        for h16 in &r16.hits {
            if let Some(&s32) = by_id_32.get(h16.chunk_id.as_str()) {
                let d = (h16.score - s32).abs();
                max_drift = max_drift.max(d);
                sum_drift += d as f64;
                drift_samples += 1;
            }
        }
    }

    let recall = total_recall / n_queries as f64;
    let mean_drift = sum_drift / drift_samples.max(1) as f64;
    eprintln!(
        "fp16 vs f32: recall@10={:.4}  drift_max={:.6}  drift_mean={:.6}",
        recall, max_drift, mean_drift
    );
    assert!(
        recall >= 0.98,
        "recall@10 too low: {:.4} (expected >= 0.98). \
         fp16 quantization or SIMD f32 accumulation regressed?",
        recall
    );
    assert!(
        max_drift <= 1e-3,
        "score drift too large: {:.6} (expected <= 1e-3). \
         fp16 SIMD path may be accumulating in fp16 instead of f32.",
        max_drift
    );

    // Cleanup.
    let _ = std::fs::remove_file(&f32_path);
    let _ = std::fs::remove_file(&f16_path);
}

#[test]
fn fp16_recall_at_1_geq_095_vs_f32() {
    // recall@1 is harder: the literal nearest neighbor must coincide
    // 95% of the time. fp16 can occasionally swap with the runner-up
    // when their scores agree to ~5 decimals.
    let n = 1000;
    let dim = 64;
    let n_queries = 50;
    let corpus = random_l2(n, dim, 0xCAFE_BABE);
    let queries = random_l2(n_queries, dim, 0xFACE_FEED);

    let f32_path = write_urna(EmbeddingDType::Float32, &corpus, dim as u32);
    let f16_path = write_urna(EmbeddingDType::Float16, &corpus, dim as u32);
    let f32_db = MmapUrnaFile::open(&f32_path).unwrap();
    let f16_db = MmapUrnaFile::open(&f16_path).unwrap();

    let mut hits = 0;
    for q in &queries {
        let r32 = f32_db.search(q, 1).unwrap();
        let r16 = f16_db.search(q, 1).unwrap();
        if r32.hits[0].chunk_id == r16.hits[0].chunk_id {
            hits += 1;
        }
    }
    let recall = hits as f64 / n_queries as f64;
    assert!(
        recall >= 0.95,
        "recall@1 too low: {:.4} (expected >= 0.95)",
        recall
    );

    let _ = std::fs::remove_file(&f32_path);
    let _ = std::fs::remove_file(&f16_path);
}
