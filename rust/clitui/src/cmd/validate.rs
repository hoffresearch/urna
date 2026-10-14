//! `urna validate <file>` - full integrity check.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use urna_format::layout::{SECTION_BLOB_DATA, SECTION_BLOB_REFS};

/// What a file that passes every check reports.
pub struct Report {
    pub sections: usize,
    /// inlined blobs proven against their content_hash, when the file has any.
    pub inlined: Option<usize>,
    pub file_hash: String,
    pub content_hash: String,
    /// the manifest's producer; files written before the field have none.
    pub producer: Option<String>,
}

/// Every check `urna validate` runs, with nothing printed: the first one
/// that fails is the error.
pub fn check(file: &Path) -> Result<Report> {
    let data = std::fs::read(file)?;
    let view = urna_format::UrnaView::from_bytes(&data)?;
    view.validate_embeddings_values()?;
    let _contract = view.search_contract()?;
    // blob_data (0x17): prove every inlined blob against its 0x14
    // content_hash, so "self-contained" is a verified claim, not a flag.
    let mut inlined = None;
    let has = |id: u32| view.section_table.iter().any(|e| e.section_id == id);
    if has(SECTION_BLOB_DATA) {
        let refs = urna_format::decode_blob_refs(&view.decoded_section(SECTION_BLOB_REFS)?)?;
        let payload = view.get_section_data(SECTION_BLOB_DATA)?;
        let table = urna_format::decode_blob_data_table(payload)?;
        if table.entries.len() != refs.len() {
            anyhow::bail!(
                "blob_data has {} entries but blob_refs has {} records",
                table.entries.len(),
                refs.len()
            );
        }
        let data = &payload[table.data_start..];
        let mut verified = 0usize;
        for (i, (r, &(off, len))) in refs.iter().zip(&table.entries).enumerate() {
            if !r.inlined {
                continue;
            }
            let bytes = &data[off as usize..(off + len) as usize];
            if Sha256::digest(bytes)[..] != r.content_hash {
                anyhow::bail!(
                    "inlined blob {} ({}) fails its content_hash",
                    i,
                    r.original_uri
                );
            }
            verified += 1;
        }
        inlined = Some(verified);
    }
    Ok(Report {
        sections: view.section_table.len(),
        inlined,
        file_hash: view.file_hash_hex(),
        content_hash: view.content_hash_hex()?,
        producer: view.manifest.producer().map(str::to_owned),
    })
}

pub fn run(file: PathBuf) -> Result<()> {
    // every check runs before the first line is printed: a file that
    // fails any of them (an inlined blob included) never shows "OK:".
    let report = check(&file)?;
    println!("OK: {} is a valid .urna v1 file", file.display());
    println!("  Header checksum:    valid");
    println!("  Section checksums:  {} sections OK", report.sections);
    println!("  Footer hash:        valid");
    println!("  Manifest:           valid (contract enforced)");
    println!("  Required sections:  all present");
    println!("  Embedding values:   no NaN/Inf");
    if let Some(verified) = report.inlined {
        println!(
            "  Inlined blobs:      {} verified against blob_refs",
            verified
        );
    }
    println!("  File hash:          {}", report.file_hash);
    println!("  Content hash:       {}", report.content_hash);
    println!(
        "  Producer:           {}",
        report.producer.as_deref().unwrap_or("-")
    );
    Ok(())
}
