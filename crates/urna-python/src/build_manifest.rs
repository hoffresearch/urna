//! Manifest assembly for `build()`, carved out of `build_fn.rs` so the
//! entry point stays under the 300-line crate guard.

use urna_format::manifest::Manifest;

/// Assemble the build manifest. the matryoshka disclosure fields
/// (`mrl_dim`/`full_dim`) are set only when truncation is active, so a
/// non-truncated file stays byte-identical with a v1 manifest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_manifest(
    embedding_model: &str,
    embedding_dim: u32,
    n_chunks: u64,
    chunker_version: &str,
    model_hash: &str,
    title: Option<String>,
    version: Option<String>,
    created: Option<String>,
    description: Option<String>,
    authors: Option<Vec<String>>,
    license: Option<String>,
    mrl_dim: Option<u32>,
    full_dim: Option<u32>,
) -> Manifest {
    Manifest {
        embedding_model: embedding_model.to_string(),
        embedding_dim,
        n_chunks,
        chunker_version: chunker_version.to_string(),
        model_hash: model_hash.to_string(),
        title,
        version,
        created,
        description,
        authors,
        license,
        mrl_dim,
        full_dim,
        ..Default::default()
    }
}
