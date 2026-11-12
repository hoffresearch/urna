//! hnsw build throughput: `HnswIndex::build` over
//! 20k x 384 synthetic l2-normalized rows at m=16 / ef_construction=200,
//! the same knobs `docs/BENCH.md` uses against hnswlib and usearch
//! (there at 100k). one sample is one full build, so the run is minutes,
//! not seconds. the build is deterministic for the seed, which is what
//! makes a before/after comparison meaningful.
//!
//! run: `cargo bench -p urna-engine --bench hnsw_build`

#[allow(dead_code, reason = "each bench uses part of the harness")]
mod harness;

use harness::Group;
use std::hint::black_box;
use std::time::Duration;
use urna_engine::ann::HnswIndex;

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

fn main() {
    let vectors = rows(N, DIM, 0xCAFE);
    let g = Group::new("hnsw_build")
        .samples(10)
        .budget(Duration::from_secs(60))
        .warmup(Duration::from_secs(5));
    g.bench(format!("m{M}_ef{EF_CONSTRUCTION}/{N}"), || {
        HnswIndex::build(
            black_box(vectors.clone()),
            N,
            DIM,
            M,
            EF_CONSTRUCTION,
            0xDEAD_BEEF,
        )
    });
}
