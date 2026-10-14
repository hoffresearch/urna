//! `BlobRef`: a content-hash reference to an original source artifact.

use serde::{Deserialize, Serialize};

/// A reference to an original source artifact addressed by content-hash.
///
/// in the default self-contained mode `inlined=true` and the original
/// bytes live inside the .urna; in catalog mode `inlined=false` and the
/// heavy bytes stay out-of-line while this record keeps the digest,
/// uri-hint, and length so a citation can reopen and verify them later.
/// `content_hash` is the raw 32-byte sha-256 of the original bytes, so a
/// catalog citation can be proven across the reference boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    pub content_hash: [u8; 32],
    pub original_uri: String,
    pub byte_len: u64,
    pub inlined: bool,
}
