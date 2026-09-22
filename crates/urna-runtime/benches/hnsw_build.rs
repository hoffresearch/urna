//! hnsw build throughput: `HnswIndex::build` over
//! 20k x 384 synthetic l2-normalized rows at m=16 / ef_construction=200,
//! the same knobs `docs/benchmarks.md` uses against hnswlib and usearch
//! (there at 100k). one sample is one full build, so the run is minutes,
//! not seconds. the build is deterministic for the seed, which is what
//! makes a before/after comparison meaningful.
//!
//! run: `cargo bench -p urna-runtime --bench hnsw_build`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use urna_runtime::ann::HnswIndex;

const N: usize = 20_000;
const DIM: usize = 384;
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;

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

fn bench_build(c: &mut Criterion) {
    let vectors = rows(N, DIM, 0xCAFE);
    let mut g = c.benchmark_group("hnsw_build");
    g.sample_size(10)
        .measurement_time(Duration::from_secs(60))
        .warm_up_time(Duration::from_secs(5));
    g.bench_with_input(
        BenchmarkId::new(format!("m{M}_ef{EF_CONSTRUCTION}"), N),
        &N,
        |b, _| {
            b.iter(|| {
                black_box(HnswIndex::build(
                    vectors.clone(),
                    N,
                    DIM,
                    M,
                    EF_CONSTRUCTION,
                    0xDEAD_BEEF,
                ))
            })
        },
    );
    g.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
