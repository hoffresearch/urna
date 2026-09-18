//! blob_data (0x17) bridge: turn the `blob_data_paths` kwarg — one
//! optional file path per blob_refs record — into the encoded section
//! payload. Reading happens here in Rust so the shard bytes never round-
//! trip through Python objects; peak memory is ~2× the media size (the
//! read buffers plus the assembled payload), which the spec docs flag for
//! very large corpora.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use urna_format::writer::UrnaFileBuilder;

use crate::build_inputs::{parse_blob_refs, parse_blob_spans};

/// Attach the optional blob trio (0x14 refs, 0x17 inlined bytes, 0x16
/// span overlay) to the builder. All additive and content_hash-excluded;
/// the overlay must have one entry per chunk (chunk order), or the
/// runtime's span rewrite would misalign.
pub(crate) fn attach_blob_sections(
    mut builder: UrnaFileBuilder,
    blob_refs: Option<&Bound<PyList>>,
    blob_data_paths: Option<&Bound<PyList>>,
    chunk_blob_spans: Option<&Bound<PyList>>,
    n_chunks: usize,
) -> PyResult<UrnaFileBuilder> {
    if let Some(refs) = blob_refs {
        let records = parse_blob_refs(refs)?;
        // blob_data (0x17) rides on the 0x14 order: one optional path per record.
        if let Some(paths) = blob_data_paths {
            let payload = build_blob_data_payload(paths, records.len())?;
            builder = builder.blob_data(payload);
        }
        let payload = urna_format::encode_blob_refs(&records)
            .map_err(|e| PyValueError::new_err(format!("blob_refs encode: {}", e)))?;
        builder = builder.blob_refs(payload);
    } else if blob_data_paths.is_some() {
        return Err(PyValueError::new_err(
            "blob_data_paths requires blob_refs (the 0x17 table parallels the 0x14 order)",
        ));
    }
    if let Some(spans) = chunk_blob_spans {
        let entries = parse_blob_spans(spans)?;
        if entries.len() != n_chunks {
            return Err(PyValueError::new_err(format!(
                "chunk_blob_spans must have one entry per chunk ({}), got {}",
                n_chunks,
                entries.len()
            )));
        }
        let payload = urna_format::encode_blob_span_overlay(&entries)
            .map_err(|e| PyValueError::new_err(format!("blob_span_overlay encode: {}", e)))?;
        builder = builder.blob_span_overlay(payload);
    }
    Ok(builder)
}

/// Parse `blob_data_paths` (list parallel to blob_refs: str path or
/// None) and encode the 0x17 payload. `n_refs` is the blob_refs record
/// count; a length mismatch is an error, never a silent misalignment.
pub(crate) fn build_blob_data_payload(paths: &Bound<PyList>, n_refs: usize) -> PyResult<Vec<u8>> {
    if paths.len() != n_refs {
        return Err(PyValueError::new_err(format!(
            "blob_data_paths must have one entry per blob_refs record ({}), got {}",
            n_refs,
            paths.len()
        )));
    }
    let mut buffers: Vec<Option<Vec<u8>>> = Vec::with_capacity(paths.len());
    for (i, item) in paths.iter().enumerate() {
        if item.is_none() {
            buffers.push(None);
            continue;
        }
        let path: String = item
            .extract()
            .map_err(|_| PyValueError::new_err(format!("blob_data_paths[{}] is not a str", i)))?;
        let bytes = std::fs::read(&path).map_err(|e| {
            PyValueError::new_err(format!("blob_data_paths[{}] ({}): {}", i, path, e))
        })?;
        buffers.push(Some(bytes));
    }
    let slices: Vec<Option<&[u8]>> = buffers.iter().map(|b| b.as_deref()).collect();
    urna_format::encode_blob_data(&slices)
        .map_err(|e| PyValueError::new_err(format!("blob_data encode: {}", e)))
}
