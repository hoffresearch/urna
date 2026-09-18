//! Decoded section access + identity hashes (file_hash, content_hash).

use std::borrow::Cow;

use super::UrnaView;
use crate::bytes::le_u64;
use crate::encoding::{decode_dedup_map, decode_payload, decode_payload_with_dict, expand_dedup};
use crate::error::UrnaError;
use crate::layout::{
    CANONICAL_SECTIONS, SECTION_CHUNKS_CANONICAL, SECTION_DEDUP_MAP, SECTION_DICTIONARY,
    SECTION_SEARCH_CONTRACT,
};
use crate::sections::{
    SearchContract, decode_chunks_canonical, decode_search_contract, encode_chunks_canonical,
};

impl<'a> UrnaView<'a> {
    /// Logical (decoded) bytes of a section's payload. Borrows for raw
    /// encoding; copies for zstd. Float16/int8 embedding payloads are
    /// returned as-is — the runtime dispatches on `manifest.dtype`.
    ///
    /// The chunks_canonical (0x02) section gets two extra, content_hash-
    /// invariant rewrites here so its decoded bytes are byte-identical to a
    /// plain build: a dict-framed (`zstd_dict`, id 5) payload is decoded
    /// against the shared dictionary in section 0x0A, and a deduped pool is
    /// re-expanded through the back-reference array in section 0x0B. Both
    /// optional sections are excluded from content_hash, so they never move a
    /// citation; the re-expansion happens BEFORE content_hash sees the bytes.
    pub fn decoded_section(&self, section_id: u32) -> crate::Result<Cow<'a, [u8]>> {
        if section_id == SECTION_CHUNKS_CANONICAL {
            return self.decoded_chunks_canonical();
        }
        self.decoded_section_plain(section_id)
    }

    /// Decode a section that needs no dict/dedup context (everything but
    /// chunks_canonical, which `decoded_section` special-cases).
    fn decoded_section_plain(&self, section_id: u32) -> crate::Result<Cow<'a, [u8]>> {
        let entry = self.entry(section_id)?;
        let phys = self.get_section_data(section_id)?;
        decode_payload(entry.encoding, phys).map_err(|e| Self::tag_err(section_id, e))
    }

    /// Decode chunks_canonical with dict (0x0A) + dedup (0x0B) awareness,
    /// rebuilding the byte-identical canonical payload a plain build emits.
    fn decoded_chunks_canonical(&self) -> crate::Result<Cow<'a, [u8]>> {
        let entry = self.entry(SECTION_CHUNKS_CANONICAL)?;
        let phys = self.get_section_data(SECTION_CHUNKS_CANONICAL)?;
        let dict = self
            .entry(SECTION_DICTIONARY)
            .ok()
            .and_then(|_| self.get_section_data(SECTION_DICTIONARY).ok());
        let decoded = decode_payload_with_dict(entry.encoding, phys, dict)
            .map_err(|e| Self::tag_err(SECTION_CHUNKS_CANONICAL, e))?;
        // no dedup map => the decoded bytes already are the full canonical
        // payload (dict/fsst/zstd all decode byte-identically to raw).
        if self.entry(SECTION_DEDUP_MAP).is_err() {
            return Ok(decoded);
        }
        // dedup map present: the section holds only the first-seen unique
        // pool; re-expand it through the back-references to the exact original
        // ordered byte stream BEFORE content_hash sees it.
        let map_phys = self.get_section_data(SECTION_DEDUP_MAP)?;
        let back_refs =
            decode_dedup_map(map_phys).map_err(|e| Self::tag_err(SECTION_DEDUP_MAP, e))?;
        let unique = decode_chunks_canonical(&decoded, count_prefix(&decoded)?)?;
        let full = expand_dedup(&unique, &back_refs)?;
        Ok(Cow::Owned(encode_chunks_canonical(&full)?))
    }

    fn tag_err(section_id: u32, e: UrnaError) -> UrnaError {
        match e {
            UrnaError::UnsupportedSectionEncoding { encoding, .. } => {
                UrnaError::UnsupportedSectionEncoding {
                    section_id,
                    encoding,
                }
            }
            UrnaError::MalformedSectionPayload { reason, .. } => {
                UrnaError::MalformedSectionPayload { section_id, reason }
            }
            other => other,
        }
    }

    /// Decode the `search_contract` section. Already validated to agree
    /// with the manifest at construction time.
    pub fn search_contract(&self) -> crate::Result<SearchContract> {
        let bytes = self.decoded_section(SECTION_SEARCH_CONTRACT)?;
        decode_search_contract(&bytes)
    }

    /// `sha256:<hex>` of the file as written, including the footer.
    pub fn file_hash_hex(&self) -> String {
        use sha2::{Digest, Sha256};
        let h = Sha256::digest(self.data);
        format!("sha256:{}", hex::encode(h))
    }

    /// `sha256:<hex>` of the canonical sections in the order fixed by spec
    /// (see `CANONICAL_SECTIONS`). Hashes the **decoded** bytes so two
    /// files that wire-compress the same logical content (zstd vs raw)
    /// produce the same content_hash and therefore stable citations.
    /// Quantized embeddings (float16 / int8) hash their on-disk bytes —
    /// they're already the canonical representation for that precision.
    /// Optional sections (HNSW, BM25, and every reserved 0x09+ section) are
    /// NOT included, and neither is the manifest: the manifest is covered by
    /// file_hash only. So additive manifest fields (a new Option field, a
    /// `capabilities_ext` flag) move file_hash but NEVER content_hash, which
    /// is why adding a capability cannot invalidate a urna:// citation.
    pub fn content_hash_hex(&self) -> crate::Result<String> {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for (id, name) in CANONICAL_SECTIONS {
            let bytes = self.decoded_section(*id)?;
            // Domain-separate by name length + name bytes so hashes for
            // different sections cannot collide via concatenation.
            h.update((name.len() as u32).to_le_bytes());
            h.update(name.as_bytes());
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(bytes.as_ref());
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }
}

/// read the u64 entry count from a canonical section payload's 12-byte
/// prefix (u32 version + u64 count) so the unique pool can be decoded back
/// to strings. bounds-checked; never panics on a short buffer.
fn count_prefix(payload: &[u8]) -> crate::Result<usize> {
    if payload.len() < 12 {
        return Err(UrnaError::MalformedSectionPayload {
            section_id: SECTION_CHUNKS_CANONICAL,
            reason: "chunks_canonical: truncated count prefix".into(),
        });
    }
    Ok(le_u64(&payload[4..12])? as usize)
}
