//! end-to-end search benchmarks through the public api (the
//! reproducible version of the docs/benchmarks.md numbers): one `.urna` per stored dtype,
//! 20k x 384 synthetic l2-normalized rows, the same hnsw graph in every
//! file (the graph is dtype-independent; the runtime materializes f32
//! vectors from the stored section), then per query:
//!
//! - `search(q, 10)`: the exact scan over every row in the stored dtype;
//! - `search_ann(q, 10, 100)`: hnsw candidates + the mandatory exact rerank
//!   over ef=100 rows, which is where the per-row rerank cost (and the
//!   allocation-free int4 path) shows.
//!
//! files land under the os temp dir and are rebuilt on every run (the
//! build is not timed). run: `cargo bench -p urna-runtime --bench rerank`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::path::PathBuf;
use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::writer::{EmbeddingDType, UrnaFileBuilder};
use urna_runtime::MmapUrnaFile;
use urna_runtime::ann::{DEFAULT_M, HnswIndex};

const N: usize = 20_000;
const DIM: usize = 384;
const K: i32 = 10;
const EF: usize = 100;
const QUERIES: usize = 64;

fn rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v = vec![0.0f32; n * dim];
    for x in v.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    }
    for row in v.chunks_exact_mut(dim) {
        let norm = row
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn build(path: &PathBuf, vectors: &[f32], hnsw: &[u8], dtype: EmbeddingDType) {
    let manifest = Manifest {
        embedding_model: "bench".into(),
        embedding_dim: DIM as u32,
        n_chunks: N as u64,
        chunker_version: "bench/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut b = UrnaFileBuilder::new(manifest).embedding_dtype(dtype);
    for i in 0..N {
        b = b.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i}"),
            source_uri: "bench.txt".into(),
            byte_start: (i * 8) as u64,
            byte_end: ((i + 1) * 8) as u64,
            embedding: vectors[i * DIM..(i + 1) * DIM].to_vec(),
        });
    }
    b.hnsw_index(hnsw.to_vec())
        .write_to_path(path)
        .expect("write bench corpus");
}

fn bench_search(c: &mut Criterion) {
    let vectors = rows(N, DIM, 0xBEEF);
    let queries = rows(QUERIES, DIM, 0xF00D);
    // ef_construction 200 keeps the setup under a minute; recall is not
    // what this bench measures.
    let hnsw = HnswIndex::build(vectors.clone(), N, DIM, DEFAULT_M, 200, 42).to_bytes();
    let dir = std::env::temp_dir();

    let dtypes = [
        ("f32", EmbeddingDType::Float32),
        ("f16", EmbeddingDType::Float16),
        ("i8", EmbeddingDType::Int8),
        ("i4", EmbeddingDType::Int4),
    ];
    let mut exact = c.benchmark_group("search_exact");
    let mut files = Vec::new();
    for (name, dtype) in dtypes {
        let path = dir.join(format!("urna-bench-rerank-{name}.urna"));
        build(&path, &vectors, &hnsw, dtype);
        let file = MmapUrnaFile::open(&path).expect("open bench corpus");
        let mut qi = 0usize;
        exact.bench_with_input(BenchmarkId::new(name, N), &N, |b, _| {
            b.iter(|| {
                let q = &queries[(qi % QUERIES) * DIM..(qi % QUERIES + 1) * DIM];
                qi += 1;
                black_box(file.search(black_box(q), K).expect("search"))
            })
        });
        files.push((name, file, path));
    }
    exact.finish();

    let mut ann = c.benchmark_group("search_ann_rerank");
    for (name, file, _) in &files {
        let mut qi = 0usize;
        ann.bench_with_input(BenchmarkId::new(*name, EF), &EF, |b, _| {
            b.iter(|| {
                let q = &queries[(qi % QUERIES) * DIM..(qi % QUERIES + 1) * DIM];
                qi += 1;
                black_box(file.search_ann(black_box(q), K, EF).expect("search_ann"))
            })
        });
    }
    ann.finish();

    for (_, file, path) in files {
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
