//! The static embedder (potion, model2vec): a token id is a row of a
//! table, a text is the mean of its rows, l2-normalized. Empty text maps
//! to the unknown token's row, as the reference embedder does.

use crate::backend::Backend;
use crate::{InferError, Result};

pub struct StaticModel<B: Backend> {
    backend: B,
    table: Vec<f32>,
    dim: usize,
    unk: u32,
}

impl<B: Backend> std::fmt::Debug for StaticModel<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticModel")
            .field("backend", &self.backend.name())
            .field("dim", &self.dim)
            .field("rows", &(self.table.len() / self.dim.max(1)))
            .finish_non_exhaustive()
    }
}

impl<B: Backend> StaticModel<B> {
    /// `table` holds rows of `dim`; `unk` is the row an empty text takes.
    pub fn new(backend: B, table: Vec<f32>, dim: usize, unk: u32) -> Result<Self> {
        if dim == 0 || !table.len().is_multiple_of(dim) {
            return Err(InferError::Shape(format!(
                "a table of {} floats is not rows of {dim}",
                table.len()
            )));
        }
        Ok(Self {
            backend,
            table,
            dim,
            unk,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn rows(&self) -> usize {
        self.table.len() / self.dim
    }

    /// The vector for token `ids`: mean of their rows, unit norm.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let ids: &[u32] = if ids.is_empty() {
            std::slice::from_ref(&self.unk)
        } else {
            ids
        };
        let mut rows = vec![0.0f32; ids.len() * self.dim];
        self.backend
            .gather_rows(&self.table, self.dim, ids, &mut rows)?;
        let mut out = vec![0.0f32; self.dim];
        self.backend.mean_pool(&rows, self.dim, &mut out)?;
        self.backend.l2_normalize(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Native;

    #[test]
    fn embeds_the_mean_of_the_rows_and_the_unk_row_for_nothing() {
        let table = vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0];
        let m = StaticModel::new(Native::detect(), table, 2, 2).expect("model");
        let v = m.embed_ids(&[0, 1]).expect("embed");
        let s = 0.5f32.hypot(0.5);
        assert!((v[0] - 0.5 / s).abs() < 1e-6 && (v[1] - 0.5 / s).abs() < 1e-6);
        let u = m.embed_ids(&[]).expect("unk");
        assert!((u[0] - 0.6).abs() < 1e-6 && (u[1] - 0.8).abs() < 1e-6);
        assert!(m.embed_ids(&[9]).is_err());
        assert!(StaticModel::new(Native::detect(), vec![1.0; 5], 2, 0).is_err());
    }
}
