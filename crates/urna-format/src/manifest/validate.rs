//! Manifest validation. Single-purpose: turn the data shape into a
//! "valid for v1 contract or not" answer with typed errors.

use super::Manifest;
use crate::error::UrnaError;
use crate::layout::{URNA_FORMAT_VERSION, URNA_SCHEMA_VERSION};

/// Default embedding dtype; v1.0 ships float32 as the recall-max baseline.
/// Compressed/quantized presets declare `dtype = "float16"` or `"int8"`.
pub const SUPPORTED_DTYPE: &str = "float32";
pub const SUPPORTED_METRIC: &str = "ip";
pub const SUPPORTED_SCORE_TYPE: &str = "cosine";
pub const SUPPORTED_NORMALIZE: &str = "l2";
pub const SUPPORTED_INDEX_TYPE: &str = "exact";
pub const SUPPORTED_RERANK_POLICY: &str = "none";

/// Every dtype the reader understands for the embeddings section.
pub const ALLOWED_DTYPES: &[&str] = &["float32", "float16", "int8", "int4"];
/// Every index_type the reader understands for the search path.
pub const ALLOWED_INDEX_TYPES: &[&str] = &["exact", "hnsw", "hybrid"];
/// Every rerank policy the reader understands. `exact` means an ANN/BM25
/// candidate set is rescored with real cosine before returning.
pub const ALLOWED_RERANK_POLICIES: &[&str] = &["none", "exact"];
/// Every score_type the reader understands.
pub const ALLOWED_SCORE_TYPES: &[&str] = &["cosine", "hybrid_rrf"];

impl Manifest {
    /// Validate that this manifest matches the v1 contract. Reject any
    /// unsupported value with a typed error rather than a string blob.
    ///
    /// Version skew policy:
    ///   - `format_version` > reader's supported version → reject (the
    ///     binary container may have changed in incompatible ways).
    ///   - `schema_version` > reader's supported version → reject (new
    ///     required fields the reader cannot enforce).
    ///   - Lower or equal versions are accepted, on the assumption that
    ///     the reader knows how to interpret older manifests.
    pub fn validate(&self) -> crate::Result<()> {
        if self.format_version > URNA_FORMAT_VERSION {
            return Err(UrnaError::UnsupportedFormatVersion(self.format_version));
        }
        if self.schema_version > URNA_SCHEMA_VERSION {
            return Err(UrnaError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.embedding_model.is_empty() {
            return Err(UrnaError::ManifestInvalid(
                "embedding_model must not be empty".into(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(UrnaError::ManifestInvalid(
                "embedding_dim must be > 0".into(),
            ));
        }
        if self.n_chunks == 0 {
            return Err(UrnaError::ManifestInvalid("n_chunks must be > 0".into()));
        }
        if self.chunker_version.is_empty() {
            return Err(UrnaError::ManifestInvalid(
                "chunker_version must not be empty".into(),
            ));
        }
        validate_model_hash(&self.model_hash)?;

        if !ALLOWED_DTYPES.contains(&self.dtype.as_str()) {
            return Err(UrnaError::UnsupportedDType(self.dtype.clone()));
        }
        if self.metric != SUPPORTED_METRIC {
            return Err(UrnaError::UnsupportedMetric(self.metric.clone()));
        }
        if !ALLOWED_SCORE_TYPES.contains(&self.score_type.as_str()) {
            return Err(UrnaError::UnsupportedScoreType(self.score_type.clone()));
        }
        if self.normalize != SUPPORTED_NORMALIZE {
            return Err(UrnaError::UnsupportedNormalize(self.normalize.clone()));
        }
        if !ALLOWED_INDEX_TYPES.contains(&self.index_type.as_str()) {
            return Err(UrnaError::UnsupportedIndexType(self.index_type.clone()));
        }
        if !ALLOWED_RERANK_POLICIES.contains(&self.rerank_policy.as_str()) {
            return Err(UrnaError::UnsupportedRerankPolicy(
                self.rerank_policy.clone(),
            ));
        }
        // ANN/hybrid index_types must declare an exact rerank so the final
        // score remains the real cosine value the user can trust.
        if (self.index_type == "hnsw" || self.index_type == "hybrid")
            && self.rerank_policy != "exact"
        {
            return Err(UrnaError::ManifestInvalid(format!(
                "index_type={} requires rerank_policy=\"exact\"",
                self.index_type
            )));
        }
        if !self.capabilities.supports_exact {
            return Err(UrnaError::ManifestInvalid(
                "capabilities.supports_exact must be true (exact path is the ground truth)".into(),
            ));
        }
        if !self.capabilities.supports_reproducible_build {
            return Err(UrnaError::ManifestInvalid(
                "capabilities.supports_reproducible_build must be true".into(),
            ));
        }
        if self.index_type == "hnsw" && !self.capabilities.supports_ann {
            return Err(UrnaError::ManifestInvalid(
                "index_type=hnsw requires capabilities.supports_ann=true".into(),
            ));
        }
        if self.index_type == "hybrid" && !self.capabilities.supports_bm25 {
            return Err(UrnaError::ManifestInvalid(
                "index_type=hybrid requires capabilities.supports_bm25=true".into(),
            ));
        }
        self.validate_matryoshka()?;
        Ok(())
    }

    /// Matryoshka disclosure invariants. `mrl_dim`/`full_dim` are additive
    /// optional; when present they must be internally consistent:
    ///   - `mrl_dim` must be > 0 (a zero prefix carries no signal),
    ///   - `mrl_dim` must be <= `full_dim` (it is a prefix of the source dim),
    ///   - `mrl_dim` must equal `embedding_dim` (the runtime strides the
    ///     embeddings section exclusively by `embedding_dim`, so the effective
    ///     stored dim IS the prefix dim),
    ///   - if `mrl_dim` is set then `full_dim` must be set too (the source dim
    ///     is the disclosure half that makes the truncation honest),
    ///   - int4 needs the EFFECTIVE dim divisible by 64 (per-64-dim block
    ///     scales), mirroring the build-time guard, so a truncated int4 file
    ///     at mrl_dim%64!=0 (e.g. 96) is rejected.
    fn validate_matryoshka(&self) -> crate::Result<()> {
        match (self.mrl_dim, self.full_dim) {
            (None, None) => Ok(()),
            (Some(0), _) => Err(UrnaError::ManifestInvalid(
                "mrl_dim must be > 0 when set".into(),
            )),
            (Some(_), None) => Err(UrnaError::ManifestInvalid(
                "mrl_dim requires full_dim (the source dim) to be set".into(),
            )),
            (None, Some(_)) => Err(UrnaError::ManifestInvalid(
                "full_dim requires mrl_dim to be set".into(),
            )),
            (Some(m), Some(f)) => {
                if m > f {
                    return Err(UrnaError::ManifestInvalid(format!(
                        "mrl_dim ({}) must be <= full_dim ({})",
                        m, f
                    )));
                }
                if m != self.embedding_dim {
                    return Err(UrnaError::ManifestInvalid(format!(
                        "mrl_dim ({}) must equal embedding_dim ({}); the runtime \
                         strides by embedding_dim",
                        m, self.embedding_dim
                    )));
                }
                if self.dtype == "int4" && m % 64 != 0 {
                    return Err(UrnaError::ManifestInvalid(format!(
                        "int4 requires mrl_dim divisible by 64, got {}",
                        m
                    )));
                }
                Ok(())
            }
        }
    }
}

/// A model_hash must be of the form `sha256:<64 hex chars>`. Anything else
/// is a contract violation: we want every claim about provenance to be
/// machine-verifiable.
fn validate_model_hash(s: &str) -> crate::Result<()> {
    let rest = s.strip_prefix("sha256:").ok_or_else(|| {
        UrnaError::InvalidModelHash(format!("expected 'sha256:<hex>' prefix, got '{}'", s))
    })?;
    if rest.len() != 64 || !rest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(UrnaError::InvalidModelHash(format!(
            "expected 64 hex chars after 'sha256:', got '{}'",
            rest
        )));
    }
    Ok(())
}

#[cfg(test)]
mod mrl_tests {
    use super::Manifest;
    use crate::error::UrnaError;

    fn good() -> Manifest {
        Manifest {
            embedding_model: "demo".into(),
            embedding_dim: 128,
            n_chunks: 1,
            chunker_version: "demo-chunker/1".into(),
            model_hash: format!("sha256:{}", "0".repeat(64)),
            ..Default::default()
        }
    }

    #[test]
    fn accepts_consistent_mrl_fields() {
        let mut m = good();
        m.embedding_dim = 128;
        m.mrl_dim = Some(128);
        m.full_dim = Some(384);
        m.validate().unwrap();
    }

    #[test]
    fn rejects_mrl_dim_greater_than_full_dim() {
        let mut m = good();
        m.embedding_dim = 512;
        m.mrl_dim = Some(512);
        m.full_dim = Some(384);
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_zero_mrl_dim() {
        let mut m = good();
        m.mrl_dim = Some(0);
        m.full_dim = Some(384);
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_mrl_dim_not_equal_embedding_dim() {
        let mut m = good();
        m.embedding_dim = 128;
        m.mrl_dim = Some(96);
        m.full_dim = Some(384);
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_mrl_dim_without_full_dim() {
        let mut m = good();
        m.mrl_dim = Some(128);
        m.full_dim = None;
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_full_dim_without_mrl_dim() {
        let mut m = good();
        m.mrl_dim = None;
        m.full_dim = Some(384);
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn rejects_int4_when_mrl_dim_not_multiple_of_64() {
        let mut m = good();
        m.embedding_dim = 96;
        m.mrl_dim = Some(96);
        m.full_dim = Some(384);
        m.dtype = "int4".into();
        assert!(matches!(m.validate(), Err(UrnaError::ManifestInvalid(_))));
    }

    #[test]
    fn accepts_int4_when_mrl_dim_multiple_of_64() {
        let mut m = good();
        m.embedding_dim = 128;
        m.mrl_dim = Some(128);
        m.full_dim = Some(384);
        m.dtype = "int4".into();
        m.validate().unwrap();
    }
}
