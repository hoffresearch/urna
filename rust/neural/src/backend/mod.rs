//! The operations a model graph is written against. A backend implements
//! them for one target; the graph never knows which. Every operation has
//! its signature here so the next family slots in without changing the
//! trait's shape, but a backend only implements what a shipped model
//! uses: the default body refuses with `Unsupported`, never silently
//! computes something else.

mod native;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;

use crate::{InferError, Result};

pub use native::Native;

/// The operations of the runtime, by name, for `Unsupported`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Matmul,
    GatherRows,
    LayerNorm,
    RmsNorm,
    Softmax,
    Attention,
    Rope,
    Gelu,
    Swiglu,
    PatchEmbed,
    MeanPool,
    L2Normalize,
}

/// A backend: one implementation of the operations for one target.
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Copy row `ids[i]` of `table` (rows of `dim`) into `out[i * dim..]`.
    /// `out.len() == ids.len() * dim`; an id past the table is a shape
    /// error, never a read past it.
    fn gather_rows(&self, table: &[f32], dim: usize, ids: &[u32], out: &mut [f32]) -> Result<()>;

    /// `out` = the mean over the rows of `rows` (each `dim` long).
    fn mean_pool(&self, rows: &[f32], dim: usize, out: &mut [f32]) -> Result<()>;

    /// Scale `v` to unit l2 norm; a zero vector stays zero.
    fn l2_normalize(&self, v: &mut [f32]) -> Result<()>;

    /// `out[m x n]` = `a[m x k]` · `b[k x n]`, row-major.
    fn matmul(
        &self,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _out: &mut [f32],
    ) -> Result<()> {
        Err(InferError::Unsupported(Op::Matmul))
    }

    /// LayerNorm over the last axis of each row, with `gamma` and `beta`.
    fn layer_norm(
        &self,
        _x: &mut [f32],
        _dim: usize,
        _gamma: &[f32],
        _beta: &[f32],
        _eps: f32,
    ) -> Result<()> {
        Err(InferError::Unsupported(Op::LayerNorm))
    }

    /// RMSNorm over the last axis of each row, with `gamma`.
    fn rms_norm(&self, _x: &mut [f32], _dim: usize, _gamma: &[f32], _eps: f32) -> Result<()> {
        Err(InferError::Unsupported(Op::RmsNorm))
    }

    /// Softmax over each row of `dim`.
    fn softmax(&self, _x: &mut [f32], _dim: usize) -> Result<()> {
        Err(InferError::Unsupported(Op::Softmax))
    }

    /// Scaled dot-product attention with grouped query heads: `q[t x h x d]`,
    /// `k`/`v[t x kv_h x d]`, `out[t x h x d]`, causal when asked.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        _q: &[f32],
        _k: &[f32],
        _v: &[f32],
        _heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _out: &mut [f32],
    ) -> Result<()> {
        Err(InferError::Unsupported(Op::Attention))
    }

    /// Rotary position embedding applied in place to `x[t x heads x d]`.
    fn rope(&self, _x: &mut [f32], _heads: usize, _head_dim: usize, _theta: f32) -> Result<()> {
        Err(InferError::Unsupported(Op::Rope))
    }

    fn gelu(&self, _x: &mut [f32]) -> Result<()> {
        Err(InferError::Unsupported(Op::Gelu))
    }

    /// `out` = silu(gate) * up, elementwise.
    fn swiglu(&self, _gate: &[f32], _up: &[f32], _out: &mut [f32]) -> Result<()> {
        Err(InferError::Unsupported(Op::Swiglu))
    }

    /// Patch embedding of an image `[c x h x w]` into `[patches x dim]`.
    #[allow(clippy::too_many_arguments)]
    fn patch_embed(
        &self,
        _image: &[f32],
        _channels: usize,
        _height: usize,
        _width: usize,
        _patch: usize,
        _weight: &[f32],
        _out: &mut [f32],
    ) -> Result<()> {
        Err(InferError::Unsupported(Op::PatchEmbed))
    }
}

/// The shape checks every backend shares, so a kernel only ever sees
/// slices that fit.
pub(crate) fn check_gather(table: &[f32], dim: usize, ids: &[u32], out: &[f32]) -> Result<()> {
    if dim == 0 || !table.len().is_multiple_of(dim) {
        return Err(InferError::Shape(format!(
            "table of {} floats is not rows of {dim}",
            table.len()
        )));
    }
    if out.len() != ids.len() * dim {
        return Err(InferError::Shape(format!(
            "gather output holds {} floats, {} ids x {dim} needed",
            out.len(),
            ids.len()
        )));
    }
    let rows = table.len() / dim;
    if let Some(bad) = ids.iter().find(|&&id| id as usize >= rows) {
        return Err(InferError::Shape(format!(
            "row {bad} past a table of {rows} rows"
        )));
    }
    Ok(())
}

pub(crate) fn check_mean(rows: &[f32], dim: usize, out: &[f32]) -> Result<()> {
    if dim == 0 || !rows.len().is_multiple_of(dim) || rows.is_empty() {
        return Err(InferError::Shape(format!(
            "{} floats are not one or more rows of {dim}",
            rows.len()
        )));
    }
    if out.len() != dim {
        return Err(InferError::Shape(format!(
            "mean output holds {} floats, {dim} needed",
            out.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testing {
    /// a tiny deterministic generator for parity tests: xorshift64*.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed.max(1))
        }
        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// uniform in [-1, 1).
        pub fn f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
        pub fn floats(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f32()).collect()
        }
    }
}
