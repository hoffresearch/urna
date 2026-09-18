//! `ChunkRecord`: a canonical-text record.

use serde::{Deserialize, Serialize};

/// A canonical-text record mirroring `builder.ChunkSpec` EXACTLY
/// (canonical_text, source_uri, byte_start, byte_end), so the python
/// adapter maps it 1:1 to a `ChunkSpec` and the byte spans round-trip
/// through `urna cite`.
///
/// forge-core does NOT chunk. producing these records is extraction;
/// splitting their canonical text into chunk-sized records is the python
/// adapter's call to the ONE authoritative chunker, `builder.chunk_text`.
/// keeping this struct byte-for-byte the shape of `ChunkSpec` is what lets
/// `urna.chunk_id` over a `ChunkRecord` equal the id over the matching
/// `ChunkSpec`, which a golden adapter test (forge-0b) locks.
///
/// the byte span indexes into the UTF-8 of the NORMALIZED source text;
/// extractors are responsible for producing NFC canonical text, and the
/// .fci stores it verbatim so the derived chunk_id never desyncs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub canonical_text: String,
    pub source_uri: String,
    pub byte_start: u64,
    pub byte_end: u64,
}
