//! chunk_scalars for `build()`: the rows and the field specs come in as
//! python lists of dicts, go through the one JSON path the `urna scalars`
//! verb uses, and the manifest gains the schema, dataset and media
//! declarations. Carved out of `build_fn.rs`, which stays the entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use urna_format::manifest::Manifest;
use urna_format::manifest::schema::{DatasetSource, MediaDerivation};
use urna_format::sections::encode_chunk_scalars;
use urna_format::sections::scalars::from_json::{self, FieldSpec};

fn to_json<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<T> {
    let s: String = py
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract()?;
    serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// Attach the columns and declarations to `manifest`; returns the encoded
/// 0x0D payload when rows were given.
pub(crate) fn attach(
    py: Python<'_>,
    mut manifest: Manifest,
    scalar_rows: Option<&Bound<'_, PyList>>,
    schema: Option<&Bound<'_, PyList>>,
    dataset: Option<&Bound<'_, PyDict>>,
    media: Option<&Bound<'_, PyDict>>,
    n_chunks: u64,
) -> PyResult<(Manifest, Option<Vec<u8>>)> {
    if let Some(d) = dataset {
        manifest.dataset = Some(to_json::<DatasetSource>(py, d.as_any(), "dataset")?);
    }
    if let Some(m) = media {
        manifest.media = Some(to_json::<MediaDerivation>(py, m.as_any(), "media")?);
    }
    let payload = match (scalar_rows, schema) {
        (None, None) => None,
        (Some(_), None) | (None, Some(_)) => {
            return Err(PyValueError::new_err(
                "scalar_rows and schema go together: the rows carry the values, the schema \
                 names and types the columns",
            ));
        }
        (Some(rows), Some(schema)) => {
            let specs: Vec<FieldSpec> = to_json(py, schema.as_any(), "schema")?;
            let rows: Vec<serde_json::Value> = to_json(py, rows.as_any(), "scalar_rows")?;
            if rows.len() as u64 != n_chunks {
                return Err(PyValueError::new_err(format!(
                    "scalar_rows has {} rows for {n_chunks} chunks",
                    rows.len()
                )));
            }
            let (table, fields) = from_json::build(&specs, &rows)
                .map_err(|e| PyValueError::new_err(format!("scalars: {e}")))?;
            manifest.schema = Some(fields);
            Some(
                encode_chunk_scalars(&table)
                    .map_err(|e| PyValueError::new_err(format!("scalars: {e}")))?,
            )
        }
    };
    Ok((manifest, payload))
}
