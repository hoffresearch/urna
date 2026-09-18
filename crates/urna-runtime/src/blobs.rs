//! Blob section open logic (0x14 blob_refs + 0x16 blob_span_overlay),
//! carved out of `mmap_file.rs` so that file stays under the 300-line
//! crate guard. Both sections are OPTIONAL and content_hash-excluded, and
//! open only behind the additive `blobs_present` capability, mirroring how
//! the graph (0x0C) opens behind `graph_present`.

use urna_format::UrnaError;
use urna_format::layout::{
    SECTION_BLOB_DATA, SECTION_BLOB_REFS, SECTION_BLOB_SPAN_OVERLAY, SECTION_ENCODING_RAW,
};
use urna_format::reader::UrnaView;
use urna_format::sections::{
    BLOB_REF_NONE, BlobRefRecord, OriginalSpan, decode_blob_data_table, decode_blob_refs,
    decode_blob_span_overlay,
};

use crate::error::RuntimeError;

/// Opened 0x17 blob_data: the offset table plus the ABSOLUTE file offset
/// of the first data byte, so `MmapUrnaFile::blob_bytes` can slice one
/// blob lazily off the mmap without ever copying the section.
#[derive(Clone, Debug)]
pub struct OpenBlobData {
    pub entries: Vec<(u64, u64)>,
    pub abs_data_start: usize,
}

/// Open the 0x17 blob_data offset table when the section is present.
/// The table must be RAW (never compressed — lazy slicing depends on it)
/// and must parallel the 0x14 record order, so a length mismatch against
/// `n_refs` is a typed format error.
pub(crate) fn open_blob_data(
    view: &UrnaView,
    n_refs: usize,
) -> Result<Option<OpenBlobData>, RuntimeError> {
    let Some(entry) = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_BLOB_DATA)
    else {
        return Ok(None);
    };
    if entry.encoding != SECTION_ENCODING_RAW {
        return Err(RuntimeError::Format(UrnaError::MalformedSectionPayload {
            section_id: SECTION_BLOB_DATA,
            reason: format!("blob_data must be raw, found encoding {}", entry.encoding),
        }));
    }
    let payload = view.get_section_data(SECTION_BLOB_DATA)?;
    let table = decode_blob_data_table(payload)?;
    if table.entries.len() != n_refs {
        return Err(RuntimeError::Format(UrnaError::MalformedSectionPayload {
            section_id: SECTION_BLOB_DATA,
            reason: format!(
                "blob_data has {} entries but blob_refs has {} records",
                table.entries.len(),
                n_refs
            ),
        }));
    }
    Ok(Some(OpenBlobData {
        entries: table.entries,
        abs_data_start: entry.offset as usize + table.data_start,
    }))
}

impl crate::mmap_file::MmapUrnaFile {
    /// Whether this file carries its media bytes inline (0x17 present).
    pub fn has_blob_data(&self) -> bool {
        self.blob_data.is_some()
    }

    /// Slice blob `index`'s bytes lazily off the mmap. Errors typed: no
    /// 0x17 section, an out-of-range index, or a record that is not
    /// inlined (its table entry is the (0, 0) gap).
    pub fn blob_bytes(&self, index: usize) -> Result<&[u8], RuntimeError> {
        let data = self
            .blob_data
            .as_ref()
            .ok_or(RuntimeError::BlobNotInlined { index })?;
        let &(offset, len) = data
            .entries
            .get(index)
            .ok_or(RuntimeError::BlobNotInlined { index })?;
        let inlined = self
            .blob_refs
            .as_ref()
            .and_then(|refs| refs.get(index))
            .is_some_and(|r| r.inlined);
        if !inlined {
            return Err(RuntimeError::BlobNotInlined { index });
        }
        let start = data.abs_data_start + offset as usize;
        let end = start + len as usize;
        // decode_blob_data_table bounded every entry against the section;
        // this guards the absolute math against the physical mmap too.
        self._mmap.get(start..end).ok_or_else(|| {
            RuntimeError::Format(UrnaError::MalformedSectionPayload {
                section_id: SECTION_BLOB_DATA,
                reason: format!("blob {} spans past the file", index),
            })
        })
    }
}

/// Open the blob pair. Returns the decoded 0x14 table (`None` when the
/// capability or section is absent). When the 0x16 overlay is present,
/// per-chunk spans that point into a blob REPLACE the decoded 0x03 spans
/// in place (for a media corpus those carry ordinal placeholders), so
/// cite/retrieve report the real blob-relative byte range against the
/// blob's uri; BLOB_REF_NONE entries keep the 0x03 span (legacy/text
/// chunks). a dangling blob_ref_index is a typed format error, never a
/// silent fallback.
pub(crate) fn open_blob_sections(
    view: &UrnaView,
    spans: &mut [OriginalSpan],
) -> Result<Option<Vec<BlobRefRecord>>, RuntimeError> {
    let blobs_present = view
        .manifest
        .capabilities_ext
        .as_ref()
        .and_then(|e| e.blobs_present)
        .unwrap_or(false);
    if !blobs_present {
        return Ok(None);
    }
    let has = |id: u32| view.section_table.iter().any(|e| e.section_id == id);
    let blob_refs = if has(SECTION_BLOB_REFS) {
        let bytes = view.decoded_section(SECTION_BLOB_REFS)?;
        Some(decode_blob_refs(&bytes)?)
    } else {
        None
    };
    if has(SECTION_BLOB_SPAN_OVERLAY) {
        let bytes = view.decoded_section(SECTION_BLOB_SPAN_OVERLAY)?;
        let overlay = decode_blob_span_overlay(&bytes)?;
        for (i, entry) in overlay.iter().enumerate().take(spans.len()) {
            if entry.blob_ref_index == BLOB_REF_NONE {
                continue;
            }
            let uri = blob_refs
                .as_ref()
                .and_then(|refs| refs.get(entry.blob_ref_index as usize))
                .map(|r| r.original_uri.clone())
                .ok_or_else(|| {
                    RuntimeError::Format(UrnaError::MalformedSectionPayload {
                        section_id: SECTION_BLOB_SPAN_OVERLAY,
                        reason: format!(
                            "overlay entry {} references missing blob_ref {}",
                            i, entry.blob_ref_index
                        ),
                    })
                })?;
            spans[i] = OriginalSpan {
                source_uri: uri,
                byte_start: entry.byte_start,
                byte_end: entry.byte_end,
            };
        }
    }
    Ok(blob_refs)
}
