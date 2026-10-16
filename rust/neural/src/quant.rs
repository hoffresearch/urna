//! Int8 quantization of a table, one scale per row (its absmax over 127),
//! so a row dequantizes with one multiply. Ready for a model whose table
//! is measured to gain from it; the potion's 30 MB table loads in tens of
//! milliseconds in f32, so it is not wired there until a measurement says
//! otherwise.

/// `(codes, scales)`: `codes[r * dim + i] * scales[r]` is the value.
pub fn quantize_rows_i8(rows: &[f32], dim: usize) -> (Vec<i8>, Vec<f32>) {
    let n = rows.len().checked_div(dim).unwrap_or(0);
    let mut codes = Vec::with_capacity(n * dim);
    let mut scales = Vec::with_capacity(n);
    for row in rows.chunks_exact(dim) {
        let absmax = row.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        scales.push(scale);
        codes.extend(
            row.iter()
                .map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8),
        );
    }
    (codes, scales)
}

/// The f32 row `r` back from `codes` and `scales`.
pub fn dequantize_row(codes: &[i8], scales: &[f32], dim: usize, r: usize) -> Vec<f32> {
    let s = scales[r];
    codes[r * dim..(r + 1) * dim]
        .iter()
        .map(|&c| c as f32 * s)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::Rng;

    #[test]
    fn a_row_dequantizes_within_half_a_step_of_its_absmax() {
        let mut rng = Rng::new(3);
        let dim = 256;
        let rows = rng.floats(dim * 4);
        let (codes, scales) = quantize_rows_i8(&rows, dim);
        assert_eq!((codes.len(), scales.len()), (dim * 4, 4));
        for r in 0..4 {
            let back = dequantize_row(&codes, &scales, dim, r);
            let absmax = rows[r * dim..(r + 1) * dim]
                .iter()
                .fold(0.0f32, |m, x| m.max(x.abs()));
            let step = absmax / 127.0;
            for (a, b) in back.iter().zip(&rows[r * dim..(r + 1) * dim]) {
                assert!((a - b).abs() <= step / 2.0 + 1e-6);
            }
        }
        let (codes, scales) = quantize_rows_i8(&[0.0, 0.0], 2);
        assert_eq!((codes, scales), (vec![0, 0], vec![1.0]));
    }
}
