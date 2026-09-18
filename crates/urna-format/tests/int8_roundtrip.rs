//! int8 codec round-trip coverage (`encoding=3`).
//!
//! These positive-path tests live here (not inline in `encoding/mod.rs`)
//! so the wire-codec registry source stays under the 300-line rust src
//! guard. Negative file-level paths live in `tests/negative_int8.rs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::{Int8EmbeddingsView, encode_int8_embeddings, quantize_f32_to_i8};

#[test]
fn int8_quantize_and_dequantize() {
    let v: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.25];
    let (scale, q) = quantize_f32_to_i8(&v);
    assert!(scale > 0.0);
    assert!(q.iter().any(|&x| x == 127 || x == -127));
    for (orig, &qi) in v.iter().zip(q.iter()) {
        let recon = qi as f32 * scale;
        assert!((orig - recon).abs() <= scale * 1.01);
    }
}

#[test]
fn int8_section_roundtrip() {
    let n = 4;
    let dim = 8;
    let mut emb: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        emb.extend_from_slice(&v);
    }
    let payload = encode_int8_embeddings(&emb, n, dim).unwrap();
    let view = Int8EmbeddingsView::parse(&payload, n, dim).unwrap();
    assert_eq!(view.n, n);
    assert_eq!(view.dim, dim);
    for i in 0..n {
        let scale = view.scale(i);
        let row = view.row(i);
        assert_eq!(row.len(), dim);
        let recon: Vec<f32> = row.iter().map(|&x| x as f32 * scale).collect();
        for (orig, r) in emb[i * dim..(i + 1) * dim].iter().zip(recon.iter()) {
            assert!((orig - r).abs() < 0.02, "{} vs {}", orig, r);
        }
    }
}
