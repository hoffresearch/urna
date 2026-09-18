//! Zero-copy view over a `.urna` byte slice.
//!
//! The reader does no I/O — callers pass an `&[u8]` (e.g. backed by an
//! `mmap`). Parsing validates magic, header checksum, file_size, all
//! section checksums, footer hash, manifest schema, and the presence of
//! every required section.
//!
//! Section payloads come in three encodings (`SECTION_ENCODING_*`):
//! - `raw`: the section bytes ARE the canonical payload.
//! - `zstd`: stored compressed; the reader decompresses on demand and
//!   returns an owned `Cow::Owned` buffer.
//! - `float16` / `int8`: only valid for the embeddings section; the
//!   physical bytes are also the canonical bytes (the runtime
//!   dispatches on `manifest.dtype`).
//!
//! Section checksums hash the **physical** bytes as stored.
//! `content_hash` hashes the **decoded** bytes so a zstd-compressed
//! corpus and its raw equivalent share the same content_hash (and
//! therefore the same citation URIs).

mod decode;
mod parse;
mod validate;
pub use validate::validate_slab_values;

use crate::error::UrnaError;
use crate::layout::{SectionEntry, UrnaFooter, UrnaHeader};
use crate::manifest::Manifest;

pub struct UrnaView<'a> {
    pub(super) data: &'a [u8],
    pub header: UrnaHeader,
    pub section_table: Vec<SectionEntry>,
    pub manifest: Manifest,
    pub footer: UrnaFooter,
}

impl<'a> UrnaView<'a> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn raw_bytes(&self) -> &[u8] {
        self.data
    }

    /// Look up the section table entry for `section_id`.
    pub fn entry(&self, section_id: u32) -> crate::Result<&SectionEntry> {
        self.section_table
            .iter()
            .find(|e| e.section_id == section_id)
            .ok_or(UrnaError::SectionNotFound(section_id))
    }

    /// Physical (on-disk, mmap-backed) bytes of a section's payload.
    /// Use `decoded_section` if you want the logical bytes (e.g. zstd
    /// decompressed) the chunk decoders consume.
    pub fn get_section_data(&self, section_id: u32) -> crate::Result<&'a [u8]> {
        let entry = self.entry(section_id)?;
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        Ok(&self.data[start..end])
    }
}
