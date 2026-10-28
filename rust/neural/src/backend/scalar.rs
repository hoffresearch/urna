//! The scalar reference for every kernel: plain loops, no intrinsics. The
//! native backend falls back to these on a target without the feature,
//! and the parity tests measure every vector kernel against them.

pub(super) fn gather_rows(table: &[f32], dim: usize, ids: &[u32], out: &mut [f32]) {
    for (slot, &id) in out.chunks_exact_mut(dim).zip(ids) {
        let start = id as usize * dim;
        slot.copy_from_slice(&table[start..start + dim]);
    }
}

pub(super) fn mean_pool(rows: &[f32], dim: usize, out: &mut [f32]) {
    out.fill(0.0);
    for row in rows.chunks_exact(dim) {
        for (o, x) in out.iter_mut().zip(row) {
            *o += x;
        }
    }
    let n = (rows.len() / dim) as f32;
    for o in out.iter_mut() {
        *o /= n;
    }
}

pub(super) fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
