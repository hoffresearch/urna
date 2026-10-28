//! `urna scalars FILE --rows ROWS --schema SCHEMA --out OUT`: write the
//! per-chunk columns a corpus carries beside its text into the file
//! itself, as the chunk_scalars (0x0D) section, with the schema that
//! names them, the dataset they came from and the media derivation in the
//! manifest. Additive: every existing section is copied as stored, so the
//! content_hash and every citation stay the same; `urna validate` on OUT
//! proves it. ROWS is a JSON array of objects (or an object holding one
//! under `rows` or `cards`), one per chunk ordinal, optionally bound to
//! the corpus by a top-level `content_hash`. SCHEMA is a JSON array of
//! fields: `name`, `kind`, `source` (the row key, `name` by default),
//! `numeric` for an enum whose values are numbers, and the manifest's
//! `label`, `unit`, `roles`, `prefix` and `display`. The kind `enum_map`
//! turns an object into one enum column per key, `name.key`.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use urna_engine::MmapUrnaFile;
use urna_format::layout::{SECTION_CHUNK_SCALARS, SECTION_ENCODING_RAW};
use urna_format::manifest::schema::{DatasetSource, MediaDerivation, SchemaField};
use urna_format::sections::scalars::from_json::{self as columns, FieldSpec};
use urna_format::sections::{ScalarTable, encode_chunk_scalars};

pub fn run(
    file: PathBuf,
    rows: Option<PathBuf>,
    schema: PathBuf,
    out: PathBuf,
    dataset: Option<PathBuf>,
    media: Option<PathBuf>,
) -> Result<()> {
    if same_file(&out, &file) {
        bail!("--out must be a different file: the corpus is rewritten, never edited in place");
    }
    let corpus = MmapUrnaFile::open(&file)?;
    let n = corpus.n_embeddings();
    let content_hash = corpus.content_hash().to_string();
    let specs: Vec<FieldSpec> = read_json(&schema)?;
    // with rows the section is built from them; without, the columns the
    // file already carries stay and the schema is declared again over them.
    let (sections, fields, n_columns) = match rows {
        Some(rows) => {
            let doc: serde_json::Value = read_json(&rows)?;
            let (bound, records) = columns::records(doc)?;
            if let Some(h) = bound
                && h != content_hash
            {
                bail!("the rows file belongs to {h}, the corpus is {content_hash}");
            }
            if records.len() != n {
                bail!("{} rows for a corpus of {n} chunks", records.len());
            }
            let (table, fields): (ScalarTable, Vec<SchemaField>) =
                columns::build(&specs, &records)?;
            let payload = encode_chunk_scalars(&table)?;
            let n_columns = table.columns.len();
            (
                vec![(SECTION_CHUNK_SCALARS, SECTION_ENCODING_RAW, payload)],
                fields,
                n_columns,
            )
        }
        None => {
            let Some(existing) = corpus.scalars() else {
                bail!(
                    "{} carries no chunk_scalars yet: pass --rows",
                    file.display()
                );
            };
            let fields = columns::declare(&specs, &existing.table)?;
            (Vec::new(), fields, existing.table.columns.len())
        }
    };
    let dataset: Option<DatasetSource> = dataset.map(|p| read_json(&p)).transpose()?;
    let media: Option<MediaDerivation> = media.map(|p| read_json(&p)).transpose()?;
    drop(corpus);

    let src = File::open(&file)?;
    // SAFETY: the corpus is opened read-only and never rewritten in place by
    // this tool (it refuses an `--out` that is the input, see same_file),
    // so no writer can truncate the mapping under us; the same model
    // `MmapUrnaFile` uses.
    let src = unsafe { memmap2::Mmap::map(&src)? };
    let dst =
        BufWriter::new(File::create(&out).with_context(|| format!("create {}", out.display()))?);
    let size = urna_format::add_sections(&src, dst, sections, |m| {
        let mut ext = m.capabilities_ext.take().unwrap_or_default();
        ext.scalars_present = Some(true);
        m.capabilities_ext = Some(ext);
        m.schema = Some(fields);
        if dataset.is_some() {
            m.dataset = dataset;
        }
        if media.is_some() {
            m.media = media;
        }
    })?;
    drop(src);

    let written = MmapUrnaFile::open(&out)?;
    if written.content_hash() != content_hash {
        bail!(
            "content_hash moved: {} -> {}",
            content_hash,
            written.content_hash()
        );
    }
    println!(
        "{}: {n_columns} columns over {n} chunks written to {} ({size} bytes); content_hash unchanged {content_hash}",
        file.display(),
        out.display()
    );
    Ok(())
}

/// `out` names the corpus itself: the same path, written another way
/// (relative, through a symlink) or a hard link to it. Writing there would
/// truncate the bytes the copy is still reading from the mapping.
fn same_file(out: &Path, file: &Path) -> bool {
    if out == file {
        return true;
    }
    let (Ok(a), Ok(b)) = (std::fs::metadata(out), std::fs::metadata(file)) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        a.dev() == b.dev() && a.ino() == b.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (a, b);
        std::fs::canonicalize(out).ok() == std::fs::canonicalize(file).ok()
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let bytes = if path.extension().is_some_and(|e| e == "zst") {
        zstd::decode_all(raw.as_slice()).with_context(|| format!("zstd {}", path.display()))?
    } else {
        raw
    };
    serde_json::from_slice(&bytes).with_context(|| format!("json {}", path.display()))
}
