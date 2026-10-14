//! Total-order comparators for f32 scores and distances.
//!
//! `partial_cmp(..).unwrap_or(Equal)` is NOT a total order once a NaN is in
//! the slice (NaN == everything, but everything else is still ordered), and
//! since Rust 1.81 `sort_by` panics when it detects that. A NaN can only
//! reach a sort through a corrupted file (a band or fp slab with a NaN
//! lane, a BM25 payload whose stats produce NaN), which the mutation fuzz
//! harness found on its first run. Every sort in the runtime routes
//! through these two functions: NaN is a total-order participant that
//! sorts LAST in both directions (worst score, farthest distance), so a
//! hostile value can never be ranked first and can never panic a sort.

use std::cmp::Ordering;

/// Descending by score, NaN last, then ascending by id for determinism.
#[inline]
pub(crate) fn cmp_score_desc(a: (usize, f32), b: (usize, f32)) -> Ordering {
    nan_last(a.1, b.1, |x, y| y.total_cmp(&x)).then_with(|| a.0.cmp(&b.0))
}

/// Ascending by distance, NaN last.
#[inline]
pub(crate) fn cmp_dist_asc(a: f32, b: f32) -> Ordering {
    nan_last(a, b, |x, y| x.total_cmp(&y))
}

#[inline]
fn nan_last(a: f32, b: f32, finite: impl Fn(f32, f32) -> Ordering) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => finite(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_sorts_last_and_never_panics() {
        let mut v = [
            (0, 0.5f32),
            (1, f32::NAN),
            (2, 0.9),
            (3, f32::NAN),
            (4, -1.0),
        ];
        v.sort_by(|a, b| cmp_score_desc(*a, *b));
        let ids: Vec<usize> = v.iter().map(|p| p.0).collect();
        assert_eq!(ids, [2, 0, 4, 1, 3]);
        let mut d = [f32::NAN, 2.0, f32::INFINITY, 0.0, f32::NAN];
        d.sort_by(|a, b| cmp_dist_asc(*a, *b));
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], f32::INFINITY);
        assert!(d[3].is_nan() && d[4].is_nan());
    }
}
