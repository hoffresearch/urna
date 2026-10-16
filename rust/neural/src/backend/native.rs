//! The native CPU backend: NEON on aarch64, AVX2 on x86_64 when the
//! machine has it, the scalar loops otherwise. Detection runs once at
//! construction; every call then dispatches on the chosen lane.

use super::{Backend, check_gather, check_mean, scalar};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

/// The CPU backend. `detect()` picks the best lane this machine runs;
/// `scalar()` is the reference the tests compare against.
#[derive(Clone, Copy, Debug)]
pub struct Native {
    lane: Lane,
}

impl Default for Native {
    fn default() -> Self {
        Self::detect()
    }
}

impl Native {
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self { lane: Lane::Neon };
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Self { lane: Lane::Avx2 };
            }
        }
        Self { lane: Lane::Scalar }
    }

    pub fn scalar() -> Self {
        Self { lane: Lane::Scalar }
    }
}

impl Backend for Native {
    fn name(&self) -> &'static str {
        match self.lane {
            Lane::Scalar => "native-scalar",
            #[cfg(target_arch = "aarch64")]
            Lane::Neon => "native-neon",
            #[cfg(target_arch = "x86_64")]
            Lane::Avx2 => "native-avx2",
        }
    }

    fn gather_rows(&self, table: &[f32], dim: usize, ids: &[u32], out: &mut [f32]) -> Result<()> {
        check_gather(table, dim, ids, out)?;
        // a row copy is a memcpy on every lane: the compiler vectorizes it,
        // and the bounds are the checked slice's.
        scalar::gather_rows(table, dim, ids, out);
        Ok(())
    }

    fn mean_pool(&self, rows: &[f32], dim: usize, out: &mut [f32]) -> Result<()> {
        check_mean(rows, dim, out)?;
        match self.lane {
            Lane::Scalar => scalar::mean_pool(rows, dim, out),
            // SAFETY: the lane was chosen only after the feature was
            // detected, and `check_mean` proved `rows` is whole rows of
            // `dim` and `out` is `dim` long.
            #[cfg(target_arch = "aarch64")]
            Lane::Neon => unsafe { super::neon::mean_pool(rows, dim, out) },
            // SAFETY: as above, for avx2+fma.
            #[cfg(target_arch = "x86_64")]
            Lane::Avx2 => unsafe { super::avx2::mean_pool(rows, dim, out) },
        }
        Ok(())
    }

    fn l2_normalize(&self, v: &mut [f32]) -> Result<()> {
        match self.lane {
            Lane::Scalar => scalar::l2_normalize(v),
            // SAFETY: the lane was chosen only after the feature was
            // detected; the kernel reads and writes within `v` only.
            #[cfg(target_arch = "aarch64")]
            Lane::Neon => unsafe { super::neon::l2_normalize(v) },
            // SAFETY: as above, for avx2+fma.
            #[cfg(target_arch = "x86_64")]
            Lane::Avx2 => unsafe { super::avx2::l2_normalize(v) },
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::Rng;
    use super::*;
    use crate::{InferError, Op};

    fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn the_detected_lane_matches_the_scalar_reference() {
        let fast = Native::detect();
        let slow = Native::scalar();
        let mut rng = Rng::new(7);
        for &(rows, dim) in &[(1usize, 1usize), (3, 7), (16, 256), (33, 255), (5, 1000)] {
            let table = rng.floats(rows * dim);
            let ids: Vec<u32> = (0..rows as u32).rev().collect();
            let mut a = vec![0.0; rows * dim];
            let mut b = vec![0.0; rows * dim];
            fast.gather_rows(&table, dim, &ids, &mut a).expect("gather");
            slow.gather_rows(&table, dim, &ids, &mut b).expect("gather");
            assert_eq!(a, b);
            let mut ma = vec![0.0; dim];
            let mut mb = vec![0.0; dim];
            fast.mean_pool(&a, dim, &mut ma).expect("mean");
            slow.mean_pool(&b, dim, &mut mb).expect("mean");
            assert!(close(&ma, &mb, 1e-5), "mean {rows}x{dim}");
            let mut na = ma.clone();
            let mut nb = mb.clone();
            fast.l2_normalize(&mut na).expect("l2");
            slow.l2_normalize(&mut nb).expect("l2");
            assert!(close(&na, &nb, 1e-6), "l2 {rows}x{dim}");
            let norm: f32 = na.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4 || na.iter().all(|&x| x == 0.0));
        }
        let mut zero = vec![0.0; 9];
        fast.l2_normalize(&mut zero).expect("l2 of zero");
        assert!(zero.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn shapes_are_checked_and_the_rest_is_refused() {
        let b = Native::detect();
        let table = vec![0.0; 8];
        let mut out = vec![0.0; 4];
        assert!(b.gather_rows(&table, 4, &[2], &mut out).is_err());
        assert!(b.gather_rows(&table, 3, &[0], &mut out).is_err());
        assert!(b.mean_pool(&table, 4, &mut out[..3]).is_err());
        assert!(matches!(
            b.softmax(&mut out, 4),
            Err(InferError::Unsupported(Op::Softmax))
        ));
        assert!(matches!(
            b.matmul(&table, &table, 2, 4, 2, &mut out),
            Err(InferError::Unsupported(Op::Matmul))
        ));
    }
}
