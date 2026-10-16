//! The per-chunk columns of chunk_scalars (0x0D) at query time: opened
//! behind the additive `scalars_present` capability like the graph and
//! the blobs, then narrowed by predicates into a bitset the search paths
//! apply BEFORE scoring, so a filtered search never pays for chunks it
//! will drop and never comes back short of k while kept chunks remain.

mod bitset;
mod filter;
mod query;

use urna_format::UrnaError;
use urna_format::layout::{SECTION_CHUNK_SCALARS, SECTION_ENCODING_RAW};
use urna_format::manifest::schema::SchemaField;
use urna_format::reader::UrnaView;
use urna_format::sections::{ScalarTable, decode_chunk_scalars};

use crate::error::RuntimeError;

pub use bitset::Bitset;
pub use filter::{Filter, Predicate};
pub use query::{ParsedQuery, parse};

/// The opened section with the manifest schema that names its columns.
#[derive(Clone, Debug, Default)]
pub struct Scalars {
    pub table: ScalarTable,
    pub schema: Vec<SchemaField>,
}

impl Scalars {
    /// The schema entry for a column, when the manifest describes it.
    pub fn field(&self, name: &str) -> Option<&SchemaField> {
        self.schema.iter().find(|f| f.name == name)
    }

    /// The field a search prefix names (`t` for the type line), when the
    /// schema declares one.
    pub fn field_by_prefix(&self, prefix: &str) -> Option<&SchemaField> {
        self.schema
            .iter()
            .find(|f| f.prefix.as_deref() == Some(prefix))
    }
}

/// Open chunk_scalars when the manifest declares `scalars_present` and
/// the section is there; the table must be RAW and exactly `n` rows.
pub(crate) fn open_scalars(view: &UrnaView<'_>, n: usize) -> Result<Option<Scalars>, RuntimeError> {
    let declared = view
        .manifest
        .capabilities_ext
        .as_ref()
        .and_then(|e| e.scalars_present)
        .unwrap_or(false);
    let Some(entry) = view
        .section_table
        .iter()
        .find(|e| e.section_id == SECTION_CHUNK_SCALARS)
    else {
        return Ok(None);
    };
    if !declared {
        return Ok(None);
    }
    if entry.encoding != SECTION_ENCODING_RAW {
        return Err(RuntimeError::Format(UrnaError::MalformedSectionPayload {
            section_id: SECTION_CHUNK_SCALARS,
            reason: format!(
                "chunk_scalars must be raw, found encoding {}",
                entry.encoding
            ),
        }));
    }
    let payload = view.get_section_data(SECTION_CHUNK_SCALARS)?;
    let table = decode_chunk_scalars(payload, n)?;
    let schema = view.manifest.schema.clone().unwrap_or_default();
    Ok(Some(Scalars { table, schema }))
}

impl crate::mmap_file::MmapUrnaFile {
    /// The per-chunk columns and their schema, when the file carries them.
    pub fn scalars(&self) -> Option<&Scalars> {
        self.scalars.as_ref()
    }

    /// Resolve `filter` over the file's columns into the bitset of chunks
    /// it keeps. A file without columns keeps everything for an empty
    /// filter and refuses any predicate.
    pub fn keep(&self, filter: &Filter) -> Result<Bitset, RuntimeError> {
        let n = self.n_embeddings;
        match self.scalars.as_ref() {
            Some(s) => filter.bitset(&s.table, n),
            None if filter.is_empty() => Ok(Bitset::full(n)),
            None => Err(RuntimeError::Filter(
                "this corpus carries no chunk_scalars to filter by".into(),
            )),
        }
    }
}
