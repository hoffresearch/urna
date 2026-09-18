//! `UrnaView::from_bytes` — header, section table, manifest, footer
//! parsing. Validation hooks are in `super::validate` and run after the
//! basic structure has been recognized.

use super::UrnaView;
use crate::error::UrnaError;
use crate::layout::{
    SECTION_ALIGNMENT, SectionEntry, URNA_FOOTER_SIZE, URNA_HEADER_SIZE, URNA_MAGIC,
    URNA_SECTION_ENTRY_SIZE, URNA_VERSION_MAJOR, URNA_VERSION_MINOR, UrnaFooter, UrnaHeader,
};
use crate::manifest::Manifest;

impl<'a> UrnaView<'a> {
    pub fn from_bytes(data: &'a [u8]) -> crate::Result<Self> {
        if data.len() < URNA_HEADER_SIZE + URNA_FOOTER_SIZE {
            return Err(UrnaError::FileTruncated);
        }

        // ---- header ----
        let mut header = UrnaHeader::default();
        header
            .as_bytes_mut()
            .copy_from_slice(&data[..URNA_HEADER_SIZE]);

        if &header.magic != URNA_MAGIC {
            return Err(UrnaError::MagicMismatch {
                expected: *URNA_MAGIC,
                got: header.magic,
            });
        }
        if header.version_major != URNA_VERSION_MAJOR || header.version_minor > URNA_VERSION_MINOR {
            return Err(UrnaError::UnsupportedVersion(
                header.version_major,
                header.version_minor,
            ));
        }
        header.validate_checksum()?;

        if header.file_size as usize != data.len() {
            return Err(UrnaError::FileSizeMismatch {
                expected: header.file_size,
                got: data.len() as u64,
            });
        }

        // ---- section table ----
        let section_table_offset = header.section_table_offset as usize;
        let section_table_count = header.section_table_count as usize;
        let section_table_end = section_table_offset
            .checked_add(
                section_table_count
                    .checked_mul(URNA_SECTION_ENTRY_SIZE)
                    .ok_or(UrnaError::SectionOffsetOutOfBounds {
                        section_id: 0,
                        offset: 0,
                    })?,
            )
            .ok_or(UrnaError::SectionOffsetOutOfBounds {
                section_id: 0,
                offset: 0,
            })?;
        if section_table_end > data.len().saturating_sub(URNA_FOOTER_SIZE) {
            return Err(UrnaError::FileTruncated);
        }

        let mut section_table = Vec::with_capacity(section_table_count);
        for i in 0..section_table_count {
            let off = section_table_offset + i * URNA_SECTION_ENTRY_SIZE;
            let mut entry = SectionEntry::new(0, 0, 0);
            entry
                .as_bytes_mut()
                .copy_from_slice(&data[off..off + URNA_SECTION_ENTRY_SIZE]);
            section_table.push(entry);
        }

        // ---- section bounds + alignment + encoding + checksums ----
        let body_end = data.len() - URNA_FOOTER_SIZE;
        for entry in &section_table {
            super::validate::validate_encoding_for_section(entry.section_id, entry.encoding)?;
            if entry.offset % SECTION_ALIGNMENT != 0 {
                return Err(UrnaError::SectionMisaligned {
                    section_id: entry.section_id,
                    offset: entry.offset,
                    alignment: SECTION_ALIGNMENT,
                });
            }
            let start = entry.offset as usize;
            let end = start.checked_add(entry.size as usize).ok_or(
                UrnaError::SectionOffsetOutOfBounds {
                    section_id: entry.section_id,
                    offset: entry.offset,
                },
            )?;
            if end > body_end {
                return Err(UrnaError::SectionOffsetOutOfBounds {
                    section_id: entry.section_id,
                    offset: entry.offset,
                });
            }
            entry.validate_checksum(&data[start..end])?;
        }

        // ---- manifest ----
        let manifest_offset = header.manifest_offset as usize;
        let manifest_size = header.manifest_size as usize;
        if manifest_offset
            .checked_add(manifest_size)
            .map(|e| e > body_end)
            .unwrap_or(true)
        {
            return Err(UrnaError::FileTruncated);
        }
        let manifest_data = &data[manifest_offset..manifest_offset + manifest_size];
        let manifest: Manifest = serde_json::from_slice(manifest_data)?;
        manifest.validate()?;

        // ---- footer hash ----
        let footer = UrnaFooter::from_bytes(&data[body_end..])?;
        let computed = UrnaFooter::compute_file_hash(&data[..body_end]);
        if computed != footer.file_hash {
            return Err(UrnaError::FooterHashMismatch);
        }

        // ---- coherence ----
        if manifest.embedding_dim != header.embedding_dim {
            return Err(UrnaError::ManifestInvalid(
                "manifest embedding_dim disagrees with header".into(),
            ));
        }
        if manifest.n_chunks != header.n_chunks {
            return Err(UrnaError::ManifestInvalid(
                "manifest n_chunks disagrees with header".into(),
            ));
        }

        let view = Self {
            data,
            header,
            section_table,
            manifest,
            footer,
        };

        // ---- semantic validation (delegates to super::validate) ----
        view.check_required_sections()?;
        view.validate_embeddings_layout()?;
        view.validate_space_bands()?;
        view.validate_search_contract()?;

        Ok(view)
    }
}
