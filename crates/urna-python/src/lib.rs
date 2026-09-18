//! PyO3 bindings for `urna_runtime`. Exposes `UrnaFile.open(path)`,
//! search variants, plus `build()` for emitting `.urna` files from
//! pre-embedded chunks. See `python/urna.py` for the Python wrapper.

use pyo3::prelude::*;

mod blob_data;
mod build_fn;
mod build_inputs;
mod build_manifest;
mod build_spaces;
mod chunk_id_fn;
mod retrieve_fn;
mod search_hit;
mod urna_file;

use build_fn::build;
use chunk_id_fn::chunk_id;
use retrieve_fn::RetrieveHitPy;
use search_hit::SearchHitPy;
use urna_file::UrnaFile;

#[pymodule]
fn _urna(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<UrnaFile>()?;
    m.add_class::<SearchHitPy>()?;
    m.add_class::<RetrieveHitPy>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_id, m)?)?;
    Ok(())
}
