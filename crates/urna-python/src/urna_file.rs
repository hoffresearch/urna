//! `UrnaFile` PyO3 class. Wraps `MmapUrnaFile` and exposes
//! search/inspect/validate to Python; hits are `SearchHitPy`
//! (`search_hit.rs`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::search_hit::SearchHitPy;

#[pyclass]
pub struct UrnaFile {
    pub(super) rt: urna_runtime::MmapUrnaFile,
}

#[pymethods]
impl UrnaFile {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let rt = urna_runtime::MmapUrnaFile::open(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(Self { rt })
    }

    fn search(&self, query: &Bound<PyAny>, k: i32) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search(&qvec, k)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// HNSW ANN search with exact rerank. Falls back to `search()` if
    /// the file has no HNSW section.
    fn search_ann(&self, query: &Bound<PyAny>, k: i32, ef: usize) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_ann(&qvec, k, ef)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// Graph search (exact top-ef seed -> bounded bfs over the chunk graph
    /// -> exact rerank on the union). Falls back to `search()` when no
    /// graph_adjacency section is present. The graph only generates
    /// candidates; the returned score is real cosine.
    #[pyo3(signature = (query, k, hops=1, ef=100))]
    fn search_graph(
        &self,
        query: &Bound<PyAny>,
        k: i32,
        hops: usize,
        ef: usize,
    ) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_graph(&qvec, k, hops, ef)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// Hybrid (BM25 ∪ vector → exact rerank). Falls back to `search()`
    /// when no BM25 section is present.
    fn search_hybrid(
        &self,
        query: &Bound<PyAny>,
        query_text: &str,
        k: i32,
        candidates: usize,
    ) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_hybrid(&qvec, query_text, k, candidates)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    /// Agent-native flagship: a pre-embedded query in, cited spans out.
    /// each hit's `score` IS the exact-cosine rerank value; routes by
    /// manifest capability (hnsw/hybrid/graph/exact). every hit carries the
    /// tier-1 stored canonical `text`, the verifying hashes, the stable
    /// citation_id, and the rerank-source precision marker. embed the query
    /// OFFLINE first (see python/forge/retrieve.py for the potion path).
    #[pyo3(signature = (query, k, candidates=None, hops=1, ef=100, expected_model_hash=None))]
    fn retrieve(
        &self,
        query: &Bound<PyAny>,
        k: i32,
        candidates: Option<usize>,
        hops: usize,
        ef: usize,
        expected_model_hash: Option<String>,
    ) -> PyResult<Vec<crate::retrieve_fn::RetrieveHitPy>> {
        crate::retrieve_fn::retrieve(
            &self.rt,
            query,
            k,
            candidates,
            hops,
            ef,
            expected_model_hash,
        )
    }

    /// Exact search over one named multimodal space (e.g. "vision").
    /// the query must be embedded with the model the space's model_hash
    /// fingerprints and have the space's dim: a text-tower query fails
    /// loudly instead of silently scoring the vision band. falls back to
    /// nothing: an unknown space is a typed error, never a silent text
    /// search.
    #[pyo3(signature = (name, query, k, expected_model_hash=None))]
    fn search_space(
        &self,
        name: &str,
        query: &Bound<PyAny>,
        k: i32,
        expected_model_hash: Option<String>,
    ) -> PyResult<Vec<SearchHitPy>> {
        let qvec: Vec<f32> = query
            .extract()
            .map_err(|e| PyValueError::new_err(format!("invalid query vector: {}", e)))?;
        let res = self
            .rt
            .search_space(name, &qvec, k, expected_model_hash.as_deref())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(res.hits.into_iter().map(SearchHitPy::from).collect())
    }

    #[getter]
    fn has_spaces(&self) -> bool {
        self.rt.has_spaces()
    }

    #[getter]
    fn space_names(&self) -> Vec<String> {
        self.rt
            .space_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[getter]
    fn embedding_dim(&self) -> usize {
        self.rt.embedding_dim()
    }

    #[getter]
    fn n_embeddings(&self) -> usize {
        self.rt.n_embeddings()
    }

    /// Chunk ids in file order (== source ordinal order): the stable
    /// identity for matching hits regardless of media ordering or backend.
    fn chunk_ids(&self) -> Vec<String> {
        self.rt.chunk_ids().to_vec()
    }

    /// One inlined blob's raw bytes (0x17), by blob_refs index — pulls a
    /// single asset without exporting the whole store.
    fn blob_bytes<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self
            .rt
            .blob_bytes(index)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, bytes))
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.rt.dtype().name()
    }

    #[getter]
    fn simd_backend(&self) -> &'static str {
        self.rt.simd_backend().name()
    }

    #[getter]
    fn has_ann(&self) -> bool {
        self.rt.has_ann()
    }

    #[getter]
    fn has_bm25(&self) -> bool {
        self.rt.has_bm25()
    }

    #[getter]
    fn has_graph(&self) -> bool {
        self.rt.has_graph()
    }

    #[getter]
    fn has_blobs(&self) -> bool {
        self.rt.has_blobs()
    }

    /// The blob_refs (0x14) table as a list of dicts (empty when the file
    /// has no blob capability): content_hash as "sha256:<hex>", the uri
    /// hint, original byte length, and the inlined flag.
    fn blob_refs(&self) -> Vec<pyo3::Py<PyDict>> {
        Python::attach(|py| {
            self.rt
                .blob_refs()
                .unwrap_or(&[])
                .iter()
                .map(|r| {
                    let d = PyDict::new(py);
                    let _ = d.set_item(
                        "content_hash",
                        format!("sha256:{}", hex::encode(r.content_hash)),
                    );
                    let _ = d.set_item("original_uri", &r.original_uri);
                    let _ = d.set_item("byte_len", r.byte_len);
                    let _ = d.set_item("inlined", r.inlined);
                    d.unbind()
                })
                .collect()
        })
    }

    #[getter]
    fn model_hash(&self) -> String {
        self.rt.model_hash().to_string()
    }

    #[getter]
    fn file_hash(&self) -> String {
        self.rt.file_hash().to_string()
    }

    #[getter]
    fn content_hash(&self) -> String {
        self.rt.content_hash().to_string()
    }

    /// Mirror of `urna inspect`: returns a Python dict with header,
    /// section table, manifest and hashes.
    fn inspect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self
            .rt
            .inspect_json()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        py.import("json")?.call_method1("loads", (s,))
    }

    /// Re-run reader-side validation. Returns `True` on success and
    /// raises `ValueError` (with the reader's typed error in the
    /// message) on any failure.
    fn validate(&self) -> PyResult<bool> {
        self.rt
            .revalidate()
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        Ok(true)
    }
}
