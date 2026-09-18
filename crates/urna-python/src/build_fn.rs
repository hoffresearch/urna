//! `build()` PyO3 function: emit a `.urna` from pre-embedded chunks.
//! Resolves preset shortcuts ("exact"/"compressed"/"tiny"/"hybrid")
//! into the underlying `EmbeddingDType` + `SectionEncoding` choices,
//! attaches optional HNSW / BM25 indices.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::build_inputs::{
    build_graph_payload, parse_chunks, resolve_preset, truncate_renormalize,
};

/// Build a .urna file from already-embedded chunks.
///
/// `chunks` is a list of dicts with keys:
///   - canonical_text: str
///   - source_uri: str
///   - byte_start: int
///   - byte_end: int
///   - embedding: list[float] (length == embedding_dim, L2-normalized)
///
/// Optional manifest fields (`title`, `version`, `created`, `description`,
/// `authors`, `license`) and a free-form `provenance` dict can be passed
/// as kwargs. `reproducible=True` overrides `created` so two builds with
/// identical inputs produce byte-identical output.
///
/// Preset / encoding kwargs:
///   - `preset`: one of "exact" (default), "compressed", "tiny", "hybrid",
///     "nano". "nano" is zstd text / int4 / hnsw (the first sub-int8 size
///     lever, ~2x smaller embeddings than tiny at stored precision; requires
///     `embedding_dim` divisible by 64).
///   - `text_encoding`: "raw" | "zstd" (overrides preset)
///   - `dtype`: "float32" | "float16" | "int8" | "int4" (overrides preset;
///     "int4" requires the EFFECTIVE dim divisible by 64)
///   - `mrl_dim`: optional matryoshka prefix dim. When set, each L2-normalized
///     f32 vector is sliced to the first `mrl_dim` components and re-L2-
///     normalized on the prefix (Qwen3/ST/BGE "truncate-then-renormalize"),
///     BEFORE quantization and HNSW build, so int8/int4 calibrate on the
///     shorter renormalized row. The file's header/manifest `embedding_dim`
///     becomes `mrl_dim`; the full source dim is recorded in `full_dim`.
///     Must satisfy `0 < mrl_dim <= embedding_dim`. A pure deterministic
///     slice + renorm, so byte-identical builds hold.
///   - `with_hnsw`: bool (overrides preset; default per preset)
///   - `with_bm25`: bool (overrides preset; default per preset)
///   - `with_graph`: bool (default off). Emits the additive chunk-to-chunk
///     graph_adjacency (0x0C) csr: NEXT_CHUNK edges (sequential ordinals) +
///     top-`graph_top_m` SEMANTIC edges from the hnsw level-0 graph,
///     canonically sorted. Excluded from content_hash, so citations are
///     unchanged. Builds (and discards) an hnsw index for the semantic edges
///     even when `with_hnsw` is off.
///   - `graph_top_m`: int (default 8). Max semantic edges per node.
///   - `blob_refs`: optional list of dicts for the blob_refs (0x14) table
///     (media blobs): keys `content_hash` ("sha256:<hex>" or bare hex),
///     `original_uri`, `byte_len`, `inlined`. entry order is the table
///     order the span overlay addresses. excluded from content_hash, so a
///     self-contained media corpus keeps the citations of its text twin.
///   - `blob_data_paths`: optional list parallel to `blob_refs` (str path
///     or None per record) inlining those files' bytes as blob_data (0x17)
///     for a self-contained corpus. requires `blob_refs`; excluded from
///     content_hash like the rest of the blob trio.
///   - `chunk_blob_spans`: optional list of dicts (one per chunk) for the
///     blob_span_overlay (0x16): keys `blob_ref_index` (int or None),
///     `byte_start`, `byte_end`. the runtime prefers these over 0x03 spans
///     for cite/retrieve. excluded from content_hash.
///   - `spaces`: optional list of dicts for multimodal embedding spaces
///     (0x15 + bands): keys `name`, `model_hash`, optional `dtype`
///     (default "float32"), `vectors` (one row per chunk). queries route
///     through `UrnaFile.search_space`, never the text path.
///   - `hnsw_m`, `hnsw_ef_construction`, `hnsw_seed`: HNSW knobs
#[pyfunction]
#[pyo3(signature = (
    output_path,
    embedding_model,
    embedding_dim,
    chunker_version,
    model_hash,
    chunks,
    *,
    title=None,
    version=None,
    created=None,
    description=None,
    authors=None,
    license=None,
    provenance=None,
    reproducible=false,
    preset="exact",
    text_encoding=None,
    dtype=None,
    mrl_dim=None,
    with_hnsw=None,
    with_bm25=None,
    with_graph=false,
    graph_top_m=8,
    blob_refs=None,
    blob_data_paths=None,
    chunk_blob_spans=None,
    spaces=None,
    hnsw_m=16,
    hnsw_ef_construction=400,
    hnsw_seed=42,
))]
#[allow(clippy::too_many_arguments)]
pub fn build(
    py: Python<'_>,
    output_path: &str,
    embedding_model: &str,
    embedding_dim: u32,
    chunker_version: &str,
    model_hash: &str,
    chunks: &Bound<PyList>,
    title: Option<String>,
    version: Option<String>,
    created: Option<String>,
    description: Option<String>,
    authors: Option<Vec<String>>,
    license: Option<String>,
    provenance: Option<&Bound<PyDict>>,
    reproducible: bool,
    preset: &str,
    text_encoding: Option<&str>,
    dtype: Option<&str>,
    mrl_dim: Option<u32>,
    with_hnsw: Option<bool>,
    with_bm25: Option<bool>,
    with_graph: bool,
    graph_top_m: usize,
    blob_refs: Option<&Bound<PyList>>,
    blob_data_paths: Option<&Bound<PyList>>,
    chunk_blob_spans: Option<&Bound<PyList>>,
    spaces: Option<&Bound<PyList>>,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_seed: u64,
) -> PyResult<String> {
    use urna_format::writer::{EmbeddingDType, UrnaFileBuilder};

    let n_chunks = chunks.len() as u64;
    let full_dim = embedding_dim;
    let mut chunk_inputs = parse_chunks(chunks)?;

    // Matryoshka prefix truncation (Qwen3/ST/BGE truncate-then-renormalize).
    // Slice each row to the first mrl_dim components and re-L2-normalize on
    // the prefix BEFORE quantization/HNSW so int8/int4 calibrate and the
    // graph builds on the shorter renormalized row. Pure deterministic op.
    // `embedding_dim` becomes the effective (truncated) dim from here on; the
    // source dim is preserved in `full_dim` for the disclosure metadata.
    let effective_dim = match mrl_dim {
        Some(d) => {
            if d == 0 || d > full_dim {
                return Err(PyValueError::new_err(format!(
                    "mrl_dim must satisfy 0 < mrl_dim <= embedding_dim ({}), got {}",
                    full_dim, d
                )));
            }
            if d < full_dim {
                truncate_renormalize(&mut chunk_inputs, full_dim as usize, d as usize);
            }
            d
        }
        None => full_dim,
    };
    let embedding_dim = effective_dim;

    let provenance_value = match provenance {
        Some(p) => {
            let s: String = py.import("json")?.call_method1("dumps", (p,))?.extract()?;
            serde_json::from_str(&s)
                .map_err(|e| PyValueError::new_err(format!("provenance JSON: {}", e)))?
        }
        None => serde_json::json!({}),
    };

    // Resolve preset defaults; explicit kwargs win (helper lives in
    // build_inputs so this entry point stays under the 300-line guard).
    let (text_enc, dt, default_hnsw, default_bm25) = resolve_preset(preset, text_encoding, dtype)?;
    // int4 requires the EFFECTIVE (post-truncation) embedding_dim to be a
    // multiple of the block size so every 64-dim group has its own absmax
    // scale. With matryoshka this means mrl_dim%64==0 (e.g. 256/192/128 are
    // valid, 96 is not). Fail fast with a clear message.
    if dt == EmbeddingDType::Int4 && (embedding_dim == 0 || embedding_dim % 64 != 0) {
        return Err(PyValueError::new_err(format!(
            "int4 requires (effective) embedding_dim divisible by 64, got {}",
            embedding_dim
        )));
    }
    let want_hnsw = with_hnsw.unwrap_or(default_hnsw);
    let want_bm25 = with_bm25.unwrap_or(default_bm25);

    // Disclosure metadata + manifest assembly live in build_manifest so
    // this entry point stays under the 300-line guard.
    let manifest = crate::build_manifest::build_manifest(
        embedding_model,
        embedding_dim,
        n_chunks,
        chunker_version,
        model_hash,
        title,
        version,
        created,
        description,
        authors,
        license,
        mrl_dim.map(|_| effective_dim),
        mrl_dim.map(|_| full_dim),
    );

    let mut builder = UrnaFileBuilder::new(manifest)
        .reproducible(reproducible)
        .with_provenance(provenance_value)
        .text_encoding(text_enc)
        .embedding_dtype(dt);

    // HNSW: build the index from f32 vectors (we have them in chunk_inputs
    // already). The runtime materializes f32 vectors at open time too —
    // here we use the originals so build is independent of dtype loss. The
    // index is also the source of top-m SEMANTIC edges for the optional graph,
    // so build it whenever hnsw OR the graph is wanted; only attach the hnsw
    // SECTION when hnsw is wanted. helper lives in build_inputs (300-line
    // guard).
    let n = chunk_inputs.len();
    let hnsw_index = if want_hnsw || with_graph {
        Some(crate::build_inputs::build_hnsw(
            &chunk_inputs,
            embedding_dim as usize,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_seed,
        ))
    } else {
        None
    };
    if want_hnsw {
        if let Some(idx) = hnsw_index.as_ref() {
            builder = builder.hnsw_index(idx.to_bytes());
        }
    }

    // Optional chunk-to-chunk graph (G1). NEXT_CHUNK edges (sequential
    // ordinals) + top-m SEMANTIC edges from the hnsw level-0 graph,
    // canonically sorted (byte-identical builds). additive, excluded from
    // content_hash; sets capabilities_ext.graph_present. default off.
    if with_graph {
        if let Some(payload) = build_graph_payload(hnsw_index.as_ref(), n, graph_top_m)? {
            builder = builder.graph_adjacency(payload);
        }
    }

    // Optional blob pair (0x14/0x16) for media corpora. both additive and
    // excluded from content_hash; the builder setters declare the additive
    // `blobs_present` capability. the overlay must have one entry per chunk
    // (chunk order), or the runtime's span rewrite would misalign.
    builder = crate::blob_data::attach_blob_sections(
        builder,
        blob_refs,
        blob_data_paths,
        chunk_blob_spans,
        n,
    )?;

    // Optional multimodal spaces (0x15 + one band per space). additive,
    // excluded from content_hash; sets supports_multimodal. the per-space
    // model_hash rides the table, so the runtime's per-space honesty gate
    // works exactly like the corpus-level one.
    if let Some(list) = spaces {
        builder = crate::build_spaces::attach_spaces(builder, list, n as u64)?;
    }

    if want_bm25 {
        let docs: Vec<String> = chunk_inputs
            .iter()
            .map(|c| c.canonical_text.clone())
            .collect();
        let bm = urna_runtime::bm25::Bm25Index::build(
            &docs,
            urna_runtime::bm25::DEFAULT_K1,
            urna_runtime::bm25::DEFAULT_B,
        );
        builder = builder.bm25_index(bm.to_bytes());
    }

    builder = builder.add_chunks(chunk_inputs);

    let bytes = builder
        .build_bytes()
        .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
    std::fs::write(output_path, &bytes)
        .map_err(|e| PyValueError::new_err(format!("write {}: {}", output_path, e)))?;
    Ok(output_path.to_string())
}
