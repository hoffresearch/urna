//! Rewriting a `.urna` with sections added: the additive path for a
//! corpus that already exists. Every existing section's physical bytes
//! are copied verbatim (same encoding, same checksum), so the decoded
//! canonical sections, and with them `content_hash` and every urna://
//! citation, are unchanged; only the section table, the manifest, the
//! new sections and `file_hash` differ. Streams to the writer, so a
//! gigabyte corpus is never held twice in memory.

use std::io::Write;

use sha2::{Digest, Sha256};

use crate::error::UrnaError;
use crate::layout::{
    SECTION_ALIGNMENT, SectionEntry, URNA_FOOTER_SIZE, URNA_HEADER_SIZE, URNA_SECTION_ENTRY_SIZE,
    UrnaHeader, align_up,
};
use crate::manifest::Manifest;
use crate::reader::UrnaView;

enum Payload<'a> {
    Existing(&'a [u8], [u8; 8]),
    New(Vec<u8>),
}

impl Payload<'_> {
    fn len(&self) -> u64 {
        match self {
            Self::Existing(b, _) => b.len() as u64,
            Self::New(b) => b.len() as u64,
        }
    }
}

/// a writer that hashes everything it forwards, for the footer's file_hash.
struct Hashing<W: Write> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write `src` (a whole `.urna`, as its bytes) to `dst` with `extra`
/// sections `(id, encoding, payload)` added, replacing any existing section
/// with the same id, and `patch` applied to the manifest. Existing payloads
/// are copied as stored. Returns the new file's size.
pub fn add_sections<W: Write>(
    src: &[u8],
    dst: W,
    extra: Vec<(u32, u32, Vec<u8>)>,
    patch: impl FnOnce(&mut Manifest),
) -> crate::Result<u64> {
    let view = UrnaView::from_bytes(src)?;
    let replaced: Vec<u32> = extra.iter().map(|(id, _, _)| *id).collect();
    let mut sections: Vec<(u32, u32, Payload<'_>)> = Vec::new();
    for e in &view.section_table {
        if replaced.contains(&e.section_id) {
            continue;
        }
        let bytes = view.get_section_data(e.section_id)?;
        sections.push((
            e.section_id,
            e.encoding,
            Payload::Existing(bytes, e.checksum),
        ));
    }
    for (id, encoding, payload) in extra {
        sections.push((id, encoding, Payload::New(payload)));
    }
    sections.sort_by_key(|s| s.0);
    if sections.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(UrnaError::InvalidInput(
            "add_sections: the same section id given twice".into(),
        ));
    }

    let mut manifest = view.manifest.clone();
    patch(&mut manifest);
    manifest.validate()?;
    let manifest_json = manifest.to_canonical_json()?;

    let header_size = URNA_HEADER_SIZE as u64;
    let table_size = sections.len() as u64 * URNA_SECTION_ENTRY_SIZE as u64;
    let manifest_offset = header_size + table_size;
    let manifest_size = manifest_json.len() as u64;
    let mut entries: Vec<SectionEntry> = Vec::with_capacity(sections.len());
    let mut offset = align_up(manifest_offset + manifest_size, SECTION_ALIGNMENT);
    for (id, encoding, payload) in &sections {
        let mut entry = SectionEntry::new(*id, offset, payload.len());
        entry.encoding = *encoding;
        match payload {
            Payload::Existing(_, checksum) => entry.checksum = *checksum,
            Payload::New(bytes) => entry.compute_checksum(bytes),
        }
        entries.push(entry);
        offset = align_up(offset + payload.len(), SECTION_ALIGNMENT);
    }
    let data_end = match (entries.last(), sections.last()) {
        (Some(e), Some((_, _, p))) => e.offset + p.len(),
        _ => manifest_offset + manifest_size,
    };
    let file_size = data_end + URNA_FOOTER_SIZE as u64;

    let header = UrnaHeader::new(
        view.header.embedding_dim,
        view.header.n_chunks,
        view.header.n_embeddings,
        file_size,
        header_size,
        sections.len() as u64,
        manifest_offset,
        manifest_size,
    );

    let mut out = Hashing {
        inner: dst,
        hasher: Sha256::new(),
    };
    let mut written: u64 = 0;
    out.write_all(header.as_bytes())?;
    written += header_size;
    for e in &entries {
        out.write_all(e.as_bytes())?;
    }
    written += table_size;
    out.write_all(&manifest_json)?;
    written += manifest_size;
    for (e, (_, _, payload)) in entries.iter().zip(&sections) {
        written = pad_to(&mut out, written, e.offset)?;
        match payload {
            Payload::Existing(bytes, _) => out.write_all(bytes)?,
            Payload::New(bytes) => out.write_all(bytes)?,
        }
        written += payload.len();
    }
    debug_assert_eq!(written, data_end);
    let hash = out.hasher.clone().finalize();
    let mut footer = Vec::with_capacity(URNA_FOOTER_SIZE);
    footer.extend_from_slice(&(URNA_FOOTER_SIZE as u64).to_le_bytes());
    footer.extend_from_slice(&hash);
    out.inner.write_all(&footer)?;
    out.flush()?;
    Ok(file_size)
}

fn pad_to<W: Write>(out: &mut W, written: u64, target: u64) -> crate::Result<u64> {
    if target < written {
        return Err(UrnaError::InvalidInput(
            "add_sections: layout went backwards".into(),
        ));
    }
    let zeros = [0u8; SECTION_ALIGNMENT as usize];
    let mut left = (target - written) as usize;
    while left > 0 {
        let n = left.min(zeros.len());
        out.write_all(&zeros[..n])?;
        left -= n;
    }
    Ok(target)
}
