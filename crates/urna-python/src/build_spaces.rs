//! Space kwarg parsing and band emission for `build()`, carved out of
//! `build_inputs.rs` so both files stay under the 300-line crate guard.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Parse and attach the optional `spaces` kwarg: a list of dicts, one per
/// non-text embedding space, with keys `name`, `model_hash` ("sha256:<hex>"
/// of the model that produced THESE vectors), optional `dtype`
/// ("float32"|"float16"|"int8"|"int4", default float32), and `vectors`
/// (list of n_chunks L2-normalized rows, all one dim). space_index is
/// assigned sequentially from 1 in list order (space 0 is always the
/// canonical text embeddings). emits the 0x15 table plus one fixed-stride
/// band (0x20 + index) per space, both excluded from content_hash.
pub(crate) fn attach_spaces(
    mut builder: urna_format::writer::UrnaFileBuilder,
    spaces: &Bound<PyList>,
    n_chunks: u64,
) -> PyResult<urna_format::writer::UrnaFileBuilder> {
    use urna_format::layout::{
        SECTION_ENCODING_FLOAT16, SECTION_ENCODING_INT4, SECTION_ENCODING_INT8,
        SECTION_ENCODING_RAW, SPACE_BAND_LEN,
    };
    use urna_format::{SpaceEntry, encode_space_table};
    if spaces.len() >= SPACE_BAND_LEN as usize {
        return Err(PyValueError::new_err(format!(
            "at most {} non-text spaces, got {}",
            SPACE_BAND_LEN - 1,
            spaces.len()
        )));
    }
    let mut entries = Vec::with_capacity(spaces.len());
    let mut bands: Vec<(u8, u32, Vec<u8>)> = Vec::with_capacity(spaces.len());
    for (i, item) in spaces.iter().enumerate() {
        let d: Bound<PyDict> = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("spaces[{}] is not a dict", i)))?
            .clone();
        let d = &d;
        let name: String = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err(format!("spaces[{}] missing name", i)))?
            .extract()?;
        let model_hash: String = d
            .get_item("model_hash")?
            .ok_or_else(|| PyValueError::new_err(format!("spaces[{}] missing model_hash", i)))?
            .extract()?;
        let dtype_str: String = match d.get_item("dtype")? {
            None => "float32".into(),
            Some(v) => v.extract()?,
        };
        let vectors: Vec<Vec<f32>> = d
            .get_item("vectors")?
            .ok_or_else(|| PyValueError::new_err(format!("spaces[{}] missing vectors", i)))?
            .extract()?;
        if vectors.len() as u64 != n_chunks {
            return Err(PyValueError::new_err(format!(
                "spaces[{}] has {} rows but the corpus has {} chunks",
                i,
                vectors.len(),
                n_chunks
            )));
        }
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let mut flat: Vec<f32> = Vec::with_capacity(vectors.len() * dim);
        for (r, row) in vectors.iter().enumerate() {
            if row.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "spaces[{}] row {} has dim {} but expected {}",
                    i,
                    r,
                    row.len(),
                    dim
                )));
            }
            flat.extend_from_slice(row);
        }
        let (dtype_code, encoding, payload) = match dtype_str.as_str() {
            "float32" => {
                let mut buf = Vec::with_capacity(flat.len() * 4);
                for v in &flat {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                (urna_format::SPACE_DTYPE_F32, SECTION_ENCODING_RAW, buf)
            }
            "float16" => (
                urna_format::SPACE_DTYPE_F16,
                SECTION_ENCODING_FLOAT16,
                urna_format::f32_to_f16_bytes(&flat),
            ),
            "int8" => (
                urna_format::SPACE_DTYPE_I8,
                SECTION_ENCODING_INT8,
                urna_format::encode_int8_embeddings(&flat, vectors.len(), dim)
                    .map_err(|e| PyValueError::new_err(format!("spaces[{}] int8: {}", i, e)))?,
            ),
            "int4" => (
                urna_format::SPACE_DTYPE_I4,
                SECTION_ENCODING_INT4,
                urna_format::encode_int4_embeddings(&flat, vectors.len(), dim)
                    .map_err(|e| PyValueError::new_err(format!("spaces[{}] int4: {}", i, e)))?,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "spaces[{}] unknown dtype: {} (expected float32|float16|int8|int4)",
                    i, other
                )));
            }
        };
        let space_index = (i + 1) as u8;
        entries.push(SpaceEntry {
            space_index,
            name,
            dim: dim as u32,
            dtype: dtype_code,
            model_hash,
            n_vectors: n_chunks,
        });
        bands.push((space_index, encoding, payload));
    }
    let table = encode_space_table(&entries)
        .map_err(|e| PyValueError::new_err(format!("space_table encode: {}", e)))?;
    builder = builder.space_table(table);
    for (idx, encoding, payload) in bands {
        builder = builder.space_band(idx, encoding, payload);
    }
    Ok(builder)
}
