//! `EmbeddingRequest`: the multimodal carrier. One chunk can request
//! several named-space embeddings, each with its own model fingerprint,
//! which is how one .urna comes to hold multiple embedding spaces.

use serde::{Deserialize, Serialize};

/// The named embedding space a request targets. `text` is `space[0]` and is
/// always the canonical text space; `image`/`glyph`/`symbol` are the
/// multimodal carriers a later phase routes per modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceTag {
    Text,
    Image,
    Glyph,
    Symbol,
}

/// What a request embeds: the linked chunk's canonical text, or an
/// external blob addressed by its raw 32-byte content-hash (an image, a
/// font glyph sheet, a rendered symbol).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadRef {
    InlineText,
    BlobHash([u8; 32]),
}

/// One named-space embedding request for one chunk (by index into the
/// bundle's `chunks`). a chunk carries several of these (text + image +
/// glyph + ...) to be embedded into distinct spaces; the determinism
/// anchor is the canonical text plus each space's `model_fingerprint`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Index into `FciBundle::chunks` this request embeds.
    pub chunk_index: u64,
    pub space: SpaceTag,
    /// The producing model's identity, `sha256:<hex>` (see
    /// python/model_fingerprint.py). recorded so a build is reproducible
    /// given the same canonical text plus the same fingerprints.
    pub model_fingerprint: String,
    pub payload_ref: PayloadRef,
}
