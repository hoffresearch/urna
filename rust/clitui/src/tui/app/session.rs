//! One corpus opened for the life of the explorer: the mmap the queries run
//! on, the stored text of every chunk, the image blobs by uri, the labels
//! the type-ahead offers and, when the file carries chunk_scalars, the
//! columns its manifest schema describes. Opening hashes the whole file,
//! so it happens once, on the loader thread, and every query after reuses
//! it.

use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use anyhow::Result;
use urna_engine::MmapUrnaFile;
use urna_engine::scalars::Scalars;
use urna_format::manifest::schema::SchemaField;
use urna_format::sections::ColumnData;

use super::labels::{Labels, alias, fold};

/// The image formats the ask tab can draw, by the blob uri's extension.
const IMAGE_EXT: [&str; 1] = [".avif"];

pub struct Session {
    pub rt: MmapUrnaFile,
    /// chunk_id -> stored canonical text (tier-1, what `cite` returns).
    pub texts: HashMap<String, String>,
    /// blob uri -> blob index, for the inlined blobs that are images.
    pub images: HashMap<String, usize>,
    pub labels: Labels,
    /// folded label and folded alias -> the chunk indices that carry it, so
    /// a query that spells a label finds it even when it ranks low.
    names: HashMap<String, Vec<usize>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("texts", &self.texts.len())
            .field("images", &self.images.len())
            .field("columns", &self.scalars().map(|s| s.table.columns.len()))
            .finish_non_exhaustive()
    }
}

impl Session {
    #[cfg(test)]
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_runtime(MmapUrnaFile::open(path)?)
    }

    pub fn from_runtime(rt: MmapUrnaFile) -> Result<Self> {
        let texts: HashMap<String, String> = rt
            .chunk_ids()
            .iter()
            .cloned()
            .zip(rt.canonical_texts()?)
            .collect();
        let images = rt
            .blob_refs()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.inlined && IMAGE_EXT.iter().any(|e| r.original_uri.ends_with(e)))
            .map(|(i, r)| (r.original_uri.clone(), i))
            .collect();
        let labels = match rt.scalars() {
            Some(s) if !label_fields(&s.schema).is_empty() => Labels::from_scalars(s),
            _ => Labels::new(
                rt.chunk_ids()
                    .iter()
                    .map(|id| texts.get(id).map_or("", String::as_str)),
            ),
        };
        let mut names: HashMap<String, Vec<usize>> = HashMap::new();
        let mut name = |key: String, i: usize| {
            let slot = names.entry(key).or_default();
            if !slot.contains(&i) {
                slot.push(i);
            }
        };
        for (i, id) in rt.chunk_ids().iter().enumerate() {
            let first = texts
                .get(id)
                .and_then(|t| t.lines().next())
                .map(str::trim)
                .unwrap_or("");
            if !first.is_empty() {
                let folded = fold(first);
                if let Some(a) = alias(&folded) {
                    name(a.to_string(), i);
                }
                name(folded, i);
            }
            if let Some(s) = rt.scalars() {
                for f in label_fields(&s.schema) {
                    if let Some(v) = text_at(s, &f.name, i) {
                        name(fold(v), i);
                    }
                }
            }
        }
        Ok(Self {
            rt,
            texts,
            images,
            labels,
            names,
        })
    }

    /// The chunk indices whose label or alias is exactly `query`, in file order.
    pub fn named(&self, query: &str) -> &[usize] {
        self.names
            .get(&fold(query.trim()))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// The image blob a hit's source points at, when it is one.
    pub fn image_of(&self, source_uri: &str) -> Option<usize> {
        self.images.get(source_uri).copied()
    }

    /// The per-chunk columns and schema, when the file carries them.
    pub fn scalars(&self) -> Option<&Scalars> {
        self.rt.scalars()
    }

    pub fn schema(&self) -> &[SchemaField] {
        self.scalars().map_or(&[], |s| s.schema.as_slice())
    }

    /// The first label-role text of chunk `i`: its name.
    pub fn label(&self, i: usize) -> Option<&str> {
        let s = self.scalars()?;
        label_fields(&s.schema)
            .into_iter()
            .find_map(|f| text_at(s, &f.name, i))
    }

    /// The rank of chunk `i` by the rank-role column (lower is more
    /// played, cited, read); `None` when unranked or without a column.
    pub fn rank(&self, i: usize) -> Option<i64> {
        let s = self.scalars()?;
        let f = s.schema.iter().find(|f| f.has_role("rank"))?;
        let c = s.table.column(&f.name)?;
        match &c.data {
            ColumnData::Int(v) if c.present(i) => v.get(i).copied(),
            ColumnData::Real(v) if c.present(i) => v.get(i).map(|x| *x as i64),
            _ => None,
        }
    }

    /// The tag-role values of chunk `i`, for a chip beside its name.
    pub fn tags(&self, i: usize) -> Vec<String> {
        let Some(s) = self.scalars() else {
            return Vec::new();
        };
        s.schema
            .iter()
            .filter(|f| f.has_role("tag"))
            .flat_map(|f| values_at(s, &f.name, i))
            .collect()
    }
}

/// One facet: the field's label and its most frequent values among the
/// kept chunks, with counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Facet {
    pub label: String,
    pub prefix: Option<String>,
    pub values: Vec<(String, usize)>,
}

impl Session {
    /// The facet-role fields counted over the chunks `keep` accepts (all
    /// when `None`), each with its `top` most frequent values.
    pub fn facets(&self, keep: Option<&urna_engine::scalars::Bitset>, top: usize) -> Vec<Facet> {
        let Some(s) = self.scalars() else {
            return Vec::new();
        };
        let n = s.table.n_rows;
        s.schema
            .iter()
            .filter(|f| f.has_role("facet"))
            .filter_map(|f| {
                let c = s.table.column(&f.name)?;
                let dict = c.data.dictionary()?;
                let mut counts = vec![0usize; dict.values.len()];
                for i in (0..n).filter(|&i| keep.is_none_or(|k| k.get(i))) {
                    if !c.present(i) {
                        continue;
                    }
                    for &code in c.data.codes_at(i) {
                        if let Some(slot) = counts.get_mut(code as usize) {
                            *slot += 1;
                        }
                    }
                }
                let mut values: Vec<(String, usize)> = dict
                    .values
                    .iter()
                    .zip(counts)
                    .filter(|(_, c)| *c > 0)
                    .map(|(v, c)| (v.clone(), c))
                    .collect();
                values.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                values.truncate(top);
                Some(Facet {
                    label: f.label.clone().unwrap_or_else(|| f.name.clone()),
                    prefix: f.prefix.clone(),
                    values,
                })
            })
            .collect()
    }
}

/// The schema fields that name a chunk, in schema order.
pub fn label_fields(schema: &[SchemaField]) -> Vec<&SchemaField> {
    schema
        .iter()
        .filter(|f| f.has_role("label") && f.kind == "text")
        .collect()
}

/// A text column's value at `row`, when present.
pub fn text_at<'a>(s: &'a Scalars, column: &str, row: usize) -> Option<&'a str> {
    let c = s.table.column(column)?;
    if !c.present(row) {
        return None;
    }
    match &c.data {
        ColumnData::Text(v) => v.get(row).map(String::as_str),
        _ => None,
    }
}

/// Every value of a column at `row` as text: one for a scalar, each entry
/// of a list, nothing when absent.
pub fn values_at(s: &Scalars, column: &str, row: usize) -> Vec<String> {
    let Some(c) = s.table.column(column) else {
        return Vec::new();
    };
    if !c.present(row) {
        return Vec::new();
    }
    match &c.data {
        ColumnData::Int(v) => v.get(row).map(|x| x.to_string()).into_iter().collect(),
        ColumnData::Real(v) => v
            .get(row)
            .map(|x| {
                if x.fract() == 0.0 {
                    format!("{}", *x as i64)
                } else {
                    x.to_string()
                }
            })
            .into_iter()
            .collect(),
        ColumnData::Bool(v) => v.get(row).map(|b| b.to_string()).into_iter().collect(),
        ColumnData::Text(v) => v
            .get(row)
            .filter(|t| !t.is_empty())
            .cloned()
            .into_iter()
            .collect(),
        ColumnData::Uuid(v) => v
            .get(row)
            .map(|u| u.iter().map(|b| format!("{b:02x}")).collect())
            .into_iter()
            .collect(),
        ColumnData::Enum { dict, .. } | ColumnData::EnumList { dict, .. } => c
            .data
            .codes_at(row)
            .iter()
            .filter_map(|&code| dict.values.get(code as usize).cloned())
            .collect(),
    }
}
