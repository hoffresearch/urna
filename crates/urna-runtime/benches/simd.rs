//! kernel benchmarks: one f32 query against one
//! stored row per dtype, at the three dims the presets use, on the backend
//! `detect_backend()` picks (the group name carries it, so a report says
//! neon / avx2 / scalar). `URNA_FORCE_SCALAR=1` benches the scalar path.
//!
//! throughput is in elements (dim) so the numbers compare across dims.
//! run: `cargo bench -p urna-runtime --bench simd`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use urna_format::{
    INT4_BLOCK, Int4EmbeddingsView, Int8EmbeddingsView, encode_int4_embeddings,
    encode_int8_embeddings, f32_to_f16_bytes,
};
use urna_runtime::simd::{
    detect_backend, dot_f32_bytes, dot_f32_f16_bytes, dot_f32_i4_blocked, dot_f32_i8,
    score_int8_section,
};

const DIMS: [usize; 3] = [256, 384, 768];
const SECTION_ROWS: usize = 50_000;

/// deterministic l2-normalized rows (lcg, no rng crate in dev-deps).
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

fn f32_le_bytes(row: &[f32]) -> Vec<u8> {
    row.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bench_dot(c: &mut Criterion) {
    let backend = detect_backend().name();
    let mut g = c.benchmark_group(format!("dot/{backend}"));
    for &dim in &DIMS {
        let data = rows(2, dim, 0x5EED);
        let (q, row) = data.split_at(dim);
        g.throughput(Throughput::Elements(dim as u64));

        let f32_bytes = f32_le_bytes(row);
        g.bench_with_input(BenchmarkId::new("f32", dim), &dim, |b, _| {
            b.iter(|| dot_f32_bytes(black_box(q), black_box(&f32_bytes)))
        });

        let f16_bytes = f32_to_f16_bytes(row);
        g.bench_with_input(BenchmarkId::new("f16", dim), &dim, |b, _| {
            b.iter(|| dot_f32_f16_bytes(black_box(q), black_box(&f16_bytes)))
        });

        let i8_payload = encode_int8_embeddings(row, 1, dim).expect("int8 encode");
        let i8_view = Int8EmbeddingsView::parse(&i8_payload, 1, dim).expect("int8 parse");
        let (i8_row, i8_scale) = (i8_view.row(0), i8_view.scale(0));
        g.bench_with_input(BenchmarkId::new("i8", dim), &dim, |b, _| {
            b.iter(|| dot_f32_i8(black_box(q), black_box(i8_row), i8_scale))
        });

        if dim % INT4_BLOCK == 0 {
            let i4_payload = encode_int4_embeddings(row, 1, dim).expect("int4 encode");
            let i4_view = Int4EmbeddingsView::parse(&i4_payload, 1, dim).expect("int4 parse");
            let codes = i4_view.row_codes(0);
            let mut scales = vec![0.0f32; dim / INT4_BLOCK];
            i4_view.row_scales_into(0, &mut scales);
            let mut scratch = vec![0.0f32; dim];
            g.bench_with_input(BenchmarkId::new("i4", dim), &dim, |b, _| {
                b.iter(|| {
                    dot_f32_i4_blocked(
                        black_box(q),
                        black_box(codes),
                        black_box(&scales),
                        dim,
                        INT4_BLOCK,
                        &mut scratch,
                    )
                })
            });
        }
    }
    g.finish();
}

/// the full int8 section scan the exact path runs: 50k rows, one query.
fn bench_section_scan(c: &mut Criterion) {
    let backend = detect_backend().name();
    let dim = 384;
    let data = rows(SECTION_ROWS + 1, dim, 0xC0DE);
    let (q, body) = data.split_at(dim);
    let payload = encode_int8_embeddings(body, SECTION_ROWS, dim).expect("int8 encode");
    let view = Int8EmbeddingsView::parse(&payload, SECTION_ROWS, dim).expect("int8 parse");
    let mut out = vec![0.0f32; SECTION_ROWS];
    let mut g = c.benchmark_group(format!("scan/{backend}"));
    g.throughput(Throughput::Elements(SECTION_ROWS as u64));
    g.bench_function(BenchmarkId::new("i8_section", SECTION_ROWS), |b| {
        b.iter(|| score_int8_section(black_box(q), black_box(&view), &mut out))
    });
    g.finish();
}

criterion_group!(benches, bench_dot, bench_section_scan);
criterion_main!(benches);
