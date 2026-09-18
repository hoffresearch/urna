//! `urna validate <file>` — full integrity check.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use urna_format::layout::{SECTION_BLOB_DATA, SECTION_BLOB_REFS};

pub fn run(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)?;
    let view = urna_format::UrnaView::from_bytes(&data)?;
    view.validate_embeddings_values()?;
    let _contract = view.search_contract()?;
    println!("OK: {} is a valid .urna v1 file", file.display());
    println!("  Header checksum:    valid");
    println!(
        "  Section checksums:  {} sections OK",
        view.section_table.len()
    );
    println!("  Footer hash:        valid");
    println!("  Manifest:           valid (contract enforced)");
    println!("  Required sections:  all present");
    println!("  Embedding values:   no NaN/Inf");
    // blob_data (0x17): prove every inlined blob against its 0x14
    // content_hash, so "self-contained" is a verified claim, not a flag.
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
        println!(
            "  Inlined blobs:      {} verified against blob_refs",
            verified
        );
    }
    println!("  File hash:          {}", view.file_hash_hex());
    println!("  Content hash:       {}", view.content_hash_hex()?);
    Ok(())
}
