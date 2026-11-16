//! `zstd_decode()` Python helper: urna-format's bounded zstd decoder, so a
//! Python tool reads a zstd frame without the `zstandard` package.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Decompress one zstd frame. The size is bounded as for a section of a
/// `.urna` (a floor of 64 MiB, else 128 times the input), so a frame that
/// would inflate past it is a `ValueError`, never an exhausted memory.
#[pyfunction]
pub fn zstd_decode<'py>(data: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyBytes>> {
    let raw = urna_format::zstd_decode(data.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("zstd_decode: {e}")))?;
    Ok(PyBytes::new(data.py(), &raw))
}
