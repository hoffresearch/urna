//! `urna benchmark <file> -q N -k K [--ann EF] [--madvise-cold]` —
//! latency stats over random queries, with optional ANN comparison +
//! recall@k vs exact, plus an opt-in madvise-cold pass.

use anyhow::Result;
use std::path::PathBuf;

pub fn run(
    file: PathBuf,
    n_queries: usize,
    k: i32,
    ann_ef: Option<usize>,
    madvise_cold: bool,
    space: Option<String>,
) -> Result<()> {
    let runtime = urna_runtime::MmapUrnaFile::open(&file)?;

    // l--space NAME: bench the named band through search_space at ITS dim.
    if let Some(name) = space {
        let data = std::fs::read(&file)?;
        let view = urna_format::UrnaView::from_bytes(&data)?;
        let payload = view.decoded_section(urna_format::layout::SECTION_SPACE_TABLE)?;
        let entry = urna_format::sections::decode_space_table(&payload)?
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow::anyhow!("space '{}' not found in the space_table", name))?;
        let queries = random_unit_queries(n_queries, entry.dim as usize);
        let times = run_bench(&runtime, &queries, false, |rt, q| {
            rt.search_space(&name, q, k, None)
        })?;
        let header = format!(
            "Space '{}' ({} queries, dim={}, dtype={})",
            name,
            n_queries,
            entry.dim,
            entry.dtype_str()
        );
        println!("{} [hot]:", header);
        print_latency(&times);
        if madvise_cold {
            let cold = run_bench(&runtime, &queries, true, |rt, q| {
                rt.search_space(&name, q, k, None)
            })?;
            println!("{} [madvise-cold]:", header);
            print_latency(&cold);
        }
        return Ok(());
    }

    let dim = runtime.embedding_dim();
    let queries = random_unit_queries(n_queries, dim);

    let header = format!(
        "Exact ({} queries, dim={}, dtype={}, simd={})",
        n_queries,
        dim,
        runtime.dtype().name(),
        runtime.simd_backend().name()
    );
    let exact_times = run_bench(&runtime, &queries, false, |rt, q| rt.search(q, k))?;
    println!("{} [hot]:", header);
    print_latency(&exact_times);

    if madvise_cold {
        let cold_times = run_bench(&runtime, &queries, true, |rt, q| rt.search(q, k))?;
        println!("{} [madvise-cold]:", header);
        print_latency(&cold_times);
        println!(
            "  (note: posix_madvise(MADV_DONTNEED) is a hint, not a guarantee. \
             Treat as an upper bound on cold-cache latency, not absolute cold.)"
        );
    }

    if let Some(ef) = ann_ef {
        if !runtime.has_ann() {
            println!("(no HNSW section — ANN bench skipped)");
            return Ok(());
        }
        let ann_times = run_bench(&runtime, &queries, false, |rt, q| rt.search_ann(q, k, ef))?;
        println!("ANN ef={} ({} queries) [hot]:", ef, n_queries);
        print_latency(&ann_times);

        if madvise_cold {
            let cold = run_bench(&runtime, &queries, true, |rt, q| rt.search_ann(q, k, ef))?;
            println!("ANN ef={} ({} queries) [madvise-cold]:", ef, n_queries);
            print_latency(&cold);
        }

        // Recall@k of ANN vs exact, computed on the same queries.
        let mut hits_overlap_total = 0.0f64;
        for q in &queries {
            let exact = runtime.search(q, k)?;
            let approx = runtime.search_ann(q, k, ef)?;
            let exact_set: std::collections::HashSet<&str> =
                exact.hits.iter().map(|h| h.chunk_id.as_str()).collect();
            let overlap = approx
                .hits
                .iter()
                .filter(|h| exact_set.contains(h.chunk_id.as_str()))
                .count();
            hits_overlap_total += overlap as f64 / k as f64;
        }
        println!(
            "  recall@{} (ANN vs exact): {:.4}",
            k,
            hits_overlap_total / n_queries as f64
        );
    }
    Ok(())
}

fn run_bench(
    rt: &urna_runtime::MmapUrnaFile,
    queries: &[Vec<f32>],
    madvise_cold: bool,
    mut f: impl FnMut(
        &urna_runtime::MmapUrnaFile,
        &[f32],
    ) -> Result<urna_runtime::SearchResult, urna_runtime::RuntimeError>,
) -> Result<Vec<f64>> {
    let mut times = Vec::with_capacity(queries.len());
    for q in queries {
        if madvise_cold {
            rt.madvise_cold();
        }
        let t0 = std::time::Instant::now();
        f(rt, q)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    Ok(times)
}

fn print_latency(times: &[f64]) {
    let p = |q: f64| -> f64 {
        let idx = ((times.len() as f64 - 1.0) * q).round() as usize;
        times[idx]
    };
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!("  mean:   {:.3} ms", mean);
    println!("  p50:    {:.3} ms", p(0.50));
    println!("  p95:    {:.3} ms", p(0.95));
    println!("  p99:    {:.3} ms", p(0.99));
}

fn random_unit_queries(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut queries = Vec::with_capacity(n);
    for _ in 0..n {
        let mut q: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut q {
                *x /= norm;
            }
        }
        queries.push(q);
    }
    queries
}
