//! Input parsing and build-time vector conditioning for `build()`.
//!
//! Kept out of `build_fn.rs` so the entry point stays under the 300-line
//! crate guard. Two helpers: `parse_chunks` (PyList of dicts ->
//! `Vec<ChunkInput>`) and `truncate_renormalize` (matryoshka prefix slice +
//! L2-renorm applied before quantization/HNSW).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Resolve the build preset plus the explicit `text_encoding`/`dtype`
/// overrides into the concrete (SectionEncoding, EmbeddingDType, hnsw,
/// bm25) quadruple. "nano" sits below "tiny": int4 block-64 embeddings at
/// stored precision (~2x over int8), zstd text, hnsw shortlist.
/// exact/compressed/tiny/hybrid are byte-frozen and unchanged.
pub(crate) fn resolve_preset(
    preset: &str,
    text_encoding: Option<&str>,
    dtype: Option<&str>,
) -> PyResult<(
    urna_format::writer::SectionEncoding,
    urna_format::writer::EmbeddingDType,
    bool,
    bool,
)> {
    use urna_format::writer::{EmbeddingDType, SectionEncoding};
    let (default_text_enc, default_dtype, default_hnsw, default_bm25) = match preset {
        "exact" => (SectionEncoding::Raw, EmbeddingDType::Float32, false, false),
        "compressed" => (SectionEncoding::Zstd, EmbeddingDType::Float16, false, false),
        "tiny" => (SectionEncoding::Zstd, EmbeddingDType::Int8, true, false),
        "nano" => (SectionEncoding::Zstd, EmbeddingDType::Int4, true, false),
        "hybrid" => (SectionEncoding::Zstd, EmbeddingDType::Float32, true, true),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown preset: {} (expected exact|compressed|tiny|nano|hybrid)",
                other
            )));
        }
    };
    let text_enc = match text_encoding {
        Some("raw") => SectionEncoding::Raw,
        Some("zstd") => SectionEncoding::Zstd,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown text_encoding: {} (expected raw|zstd)",
                other
            )));
        }
        None => default_text_enc,
    };
    let dt = match dtype {
        Some("float32") => EmbeddingDType::Float32,
        Some("float16") => EmbeddingDType::Float16,
        Some("int8") => EmbeddingDType::Int8,
        Some("int4") => EmbeddingDType::Int4,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown dtype: {} (expected float32|float16|int8|int4)",
                other
            )));
        }
        None => default_dtype,
    };
    Ok((text_enc, dt, default_hnsw, default_bm25))
}

/// Parse the optional `blob_refs` kwarg: a list of dicts with keys
/// `content_hash` ("sha256:<64 hex>" or bare 64 hex), `original_uri`,
/// `byte_len`, `inlined`. entry order is preserved: the 0x14 table is
/// addressed by ordinal from the span overlay.
pub(crate) fn parse_blob_refs(refs: &Bound<PyList>) -> PyResult<Vec<urna_format::BlobRefRecord>> {
    let mut out = Vec::with_capacity(refs.len());
    for (i, item) in refs.iter().enumerate() {
        let d: Bound<PyDict> = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("blob_refs[{}] is not a dict", i)))?
            .clone();
        let d = &d;
        let hash_str: String = d
            .get_item("content_hash")?
            .ok_or_else(|| PyValueError::new_err(format!("blob_refs[{}] missing content_hash", i)))?
            .extract()?;
        let hex_part = hash_str.strip_prefix("sha256:").unwrap_or(&hash_str);
        let raw = hex::decode(hex_part).map_err(|e| {
            PyValueError::new_err(format!("blob_refs[{}] content_hash hex: {}", i, e))
        })?;
        let content_hash: [u8; 32] = raw.try_into().map_err(|_| {
            PyValueError::new_err(format!("blob_refs[{}] content_hash must be 32 bytes", i))
        })?;
        let original_uri: String = d
            .get_item("original_uri")?
            .ok_or_else(|| PyValueError::new_err(format!("blob_refs[{}] missing original_uri", i)))?
            .extract()?;
        let byte_len: u64 = d
            .get_item("byte_len")?
            .ok_or_else(|| PyValueError::new_err(format!("blob_refs[{}] missing byte_len", i)))?
            .extract()?;
        let inlined: bool = d
            .get_item("inlined")?
            .ok_or_else(|| PyValueError::new_err(format!("blob_refs[{}] missing inlined", i)))?
            .extract()?;
        out.push(urna_format::BlobRefRecord {
            content_hash,
            original_uri,
            byte_len,
            inlined,
        });
    }
    Ok(out)
}

/// Parse the optional `chunk_blob_spans` kwarg: a list of dicts with keys
/// `blob_ref_index` (int, or None for BLOB_REF_NONE), `byte_start`,
/// `byte_end`. one entry per chunk, in chunk order.
pub(crate) fn parse_blob_spans(spans: &Bound<PyList>) -> PyResult<Vec<urna_format::BlobSpanEntry>> {
    let mut out = Vec::with_capacity(spans.len());
    for (i, item) in spans.iter().enumerate() {
        let d: Bound<PyDict> = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("chunk_blob_spans[{}] is not a dict", i)))?
            .clone();
        let d = &d;
        let blob_ref_index = match d.get_item("blob_ref_index")? {
            None => {
                return Err(PyValueError::new_err(format!(
                    "chunk_blob_spans[{}] missing blob_ref_index",
                    i
                )));
            }
            Some(v) if v.is_none() => urna_format::BLOB_REF_NONE,
            Some(v) => v.extract::<u32>()?,
        };
        let byte_start: u64 = d
            .get_item("byte_start")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("chunk_blob_spans[{}] missing byte_start", i))
            })?
            .extract()?;
        let byte_end: u64 = d
            .get_item("byte_end")?
            .ok_or_else(|| {
                PyValueError::new_err(format!("chunk_blob_spans[{}] missing byte_end", i))
            })?
            .extract()?;
        out.push(urna_format::BlobSpanEntry {
            blob_ref_index,
            byte_start,
            byte_end,
        });
    }
    Ok(out)
}

/// Truncate each row to its first `mrl_dim` components and re-L2-normalize
/// the prefix in place (matryoshka truncate-then-renormalize). `full_dim` is
/// the source dim; rows shorter than `full_dim` are left untouched (the
/// builder's per-chunk validation rejects a real dim mismatch later). A zero
/// or non-finite prefix norm leaves the (already truncated) row unscaled, so
/// the codec/validation path still sees a well-formed-length vector and
/// reports the problem with a typed error rather than producing NaNs.
pub(crate) fn truncate_renormalize(
    chunks: &mut [urna_format::ChunkInput],
    full_dim: usize,
    mrl_dim: usize,
) {
    for c in chunks.iter_mut() {
        if c.embedding.len() != full_dim {
            continue;
        }
        c.embedding.truncate(mrl_dim);
        let norm: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 && norm.is_finite() {
            for x in &mut c.embedding {
                *x /= norm;
            }
        }
    }
}

/// Build the chunk-to-chunk graph_adjacency (0x0C) csr payload for `n`
/// chunks: NEXT_CHUNK edges (sequential ordinals, both directions so the
/// bounded bfs can reconstruct neighbor context on either side) plus up to
/// `top_m` SEMANTIC edges per node taken from the already-built hnsw level-0
/// adjacency. canonically sorted by `encode_graph_adjacency` (ascending src,
/// edge_type, dst) so two builds are byte-identical. returns `None` when
/// there is nothing to emit (n < 2). deterministic and pure.
pub(crate) fn build_graph_payload(
    hnsw: Option<&urna_runtime::ann::HnswIndex>,
    n: usize,
    top_m: usize,
) -> PyResult<Option<Vec<u8>>> {
    use urna_format::{EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, Edge, encode_graph_adjacency};
    if n < 2 {
        return Ok(None);
    }
    let mut edges: Vec<Edge> = Vec::new();
    // NEXT_CHUNK: i <-> i+1 (sequential reading order, both directions).
    for i in 0..n - 1 {
        edges.push(Edge {
            src: i as u32,
            dst: (i + 1) as u32,
            edge_type: EDGE_TYPE_NEXT_CHUNK,
        });
        edges.push(Edge {
            src: (i + 1) as u32,
            dst: i as u32,
            edge_type: EDGE_TYPE_NEXT_CHUNK,
        });
    }
    // SEMANTIC: top-m from the hnsw level-0 graph (already built, no o(n^2)
    // knn). skip self-loops; the encoder dedups by canonical sort, so a node
    // appearing in both a NEXT_CHUNK and a SEMANTIC edge keeps both typed
    // edges (different edge_type = different canonical key).
    if let Some(idx) = hnsw {
        for i in 0..n {
            let mut count = 0usize;
            for &nbr in idx.level0_neighbors(i) {
                if count >= top_m {
                    break;
                }
                if nbr as usize == i {
                    continue;
                }
                edges.push(Edge {
                    src: i as u32,
                    dst: nbr,
                    edge_type: EDGE_TYPE_SEMANTIC,
                });
                count += 1;
            }
        }
    }
    let payload = encode_graph_adjacency(n, &edges)
        .map_err(|e| PyValueError::new_err(format!("graph_adjacency encode: {}", e)))?;
    Ok(Some(payload))
}

/// Build the hnsw index over the chunk embeddings (flattened f32 rows).
/// carved out of `build_fn.rs` (300-line guard): the index doubles as the
/// source of top-m SEMANTIC edges for the optional graph, so it is built
/// whenever hnsw OR the graph is wanted.
pub(crate) fn build_hnsw(
    chunks: &[urna_format::ChunkInput],
    dim: usize,
    m: usize,
    ef_construction: usize,
    seed: u64,
) -> urna_runtime::ann::HnswIndex {
    let mut flat: Vec<f32> = Vec::with_capacity(chunks.len() * dim);
    for c in chunks {
        flat.extend_from_slice(&c.embedding);
    }
    urna_runtime::ann::HnswIndex::build(flat, chunks.len(), dim, m, ef_construction, seed)
}

pub(crate) fn parse_chunks(chunks: &Bound<PyList>) -> PyResult<Vec<urna_format::ChunkInput>> {
    use urna_format::ChunkInput;
    let mut out: Vec<ChunkInput> = Vec::with_capacity(chunks.len());
    for (i, item) in chunks.iter().enumerate() {
        let d: Bound<PyDict> = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("chunks[{}] is not a dict", i)))?
            .clone();
        let d = &d;
        let canonical_text: String = d
            .get_item("canonical_text")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing canonical_text", i)))?
            .extract()?;
        let source_uri: String = d
            .get_item("source_uri")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing source_uri", i)))?
            .extract()?;
        let byte_start: u64 = d
            .get_item("byte_start")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing byte_start", i)))?
            .extract()?;
        let byte_end: u64 = d
            .get_item("byte_end")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing byte_end", i)))?
            .extract()?;
        let embedding: Vec<f32> = d
            .get_item("embedding")?
            .ok_or_else(|| PyValueError::new_err(format!("chunks[{}] missing embedding", i)))?
            .extract()?;
        out.push(ChunkInput {
            canonical_text,
            source_uri,
            byte_start,
            byte_end,
            embedding,
        });
    }
    Ok(out)
}
