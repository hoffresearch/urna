//! The benches' own harness, in place of criterion: warm up, size the
//! samples from the warm-up, time them with `Instant`, report the median
//! (p50) and the 95th percentile (p95) per iteration, and the throughput
//! when the group names its elements. No statistics past the two
//! percentiles, no report files: numbers to compare before and after a
//! change on one machine, which is all these benches are for (gatecheck
//! only builds them; shared runners are no place for timings).
//!
//! `cargo bench -p urna-engine --bench simd -- f16` runs the benchmarks
//! whose name contains `f16`; cargo's own `--bench` flag is ignored.

use std::time::{Duration, Instant};

/// A named set of benchmarks sharing one configuration.
pub struct Group {
    name: String,
    samples: usize,
    warmup: Duration,
    budget: Duration,
    elements: Option<u64>,
    filter: Option<String>,
}

impl Group {
    /// 50 samples after a one-second warm-up, about three seconds each run.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            samples: 50,
            warmup: Duration::from_secs(1),
            budget: Duration::from_secs(3),
            elements: None,
            filter: std::env::args().skip(1).find(|a| !a.starts_with("--")),
        }
    }

    pub fn samples(mut self, n: usize) -> Self {
        self.samples = n.max(2);
        self
    }

    pub fn warmup(mut self, d: Duration) -> Self {
        self.warmup = d;
        self
    }

    /// The time the samples of one benchmark aim to take together.
    pub fn budget(mut self, d: Duration) -> Self {
        self.budget = d;
        self
    }

    /// Elements one iteration processes, for an elements/s figure.
    pub fn throughput(&mut self, elements: u64) {
        self.elements = Some(elements);
    }

    /// Times `f` and prints `group/id`, p50 and p95 per iteration.
    pub fn bench<R>(&self, id: impl std::fmt::Display, mut f: impl FnMut() -> R) {
        let name = format!("{}/{id}", self.name);
        if self
            .filter
            .as_ref()
            .is_some_and(|w| !name.contains(w.as_str()))
        {
            return;
        }
        // warm-up: at least one call, then until the warm-up time is spent;
        // its mean sizes the samples.
        let start = Instant::now();
        let mut calls = 0u64;
        while calls == 0 || start.elapsed() < self.warmup {
            std::hint::black_box(f());
            calls += 1;
        }
        let mean = start.elapsed().as_secs_f64() / calls as f64;
        let per_sample = self.budget.as_secs_f64() / self.samples as f64;
        let iters = ((per_sample / mean.max(1e-9)) as u64).max(1);
        let mut times: Vec<f64> = (0..self.samples)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..iters {
                    std::hint::black_box(f());
                }
                t.elapsed().as_secs_f64() / iters as f64
            })
            .collect();
        times.sort_by(f64::total_cmp);
        let p50 = percentile(&times, 0.50);
        let p95 = percentile(&times, 0.95);
        let rate = self
            .elements
            .map(|n| format!("  {:.3} Melem/s", n as f64 / p50 / 1e6))
            .unwrap_or_default();
        println!(
            "{name:<40} p50 {:>10}  p95 {:>10}{rate}  ({} x {iters})",
            human(p50),
            human(p95),
            self.samples
        );
    }
}

/// The value at quantile `q` of sorted `xs` (nearest rank).
fn percentile(xs: &[f64], q: f64) -> f64 {
    let rank = ((xs.len() as f64 * q).ceil() as usize).clamp(1, xs.len());
    xs[rank - 1]
}

fn human(secs: f64) -> String {
    match secs {
        s if s < 1e-6 => format!("{:.1} ns", s * 1e9),
        s if s < 1e-3 => format!("{:.2} us", s * 1e6),
        s if s < 1.0 => format!("{:.2} ms", s * 1e3),
        s => format!("{s:.2} s"),
    }
}
