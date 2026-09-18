//! `urna inspect <file> [--json]` — header, section table, manifest,
//! hashes. JSON variant mirrors `MmapUrnaFile::inspect_json` for
//! programmatic consumers.

use anyhow::Result;
use std::path::PathBuf;

use super::util::encoding_name;

pub fn run(file: PathBuf, json: bool) -> Result<()> {
    if json {
        return run_json(file);
    }
    let data = std::fs::read(&file)?;
    let view = urna_format::UrnaView::from_bytes(&data)?;
    let magic = std::str::from_utf8(&view.header.magic).unwrap_or("???");
    println!("Magic:        {:?}", magic);
    println!(
        "Version:      {}.{}",
        view.header.version_major, view.header.version_minor
    );
    println!("File size:    {} bytes", view.header.file_size);
    println!("Sections:     {}", view.section_table.len());
    for entry in &view.section_table {
        let name = urna_format::layout::section_name(entry.section_id).unwrap_or("unknown");
        let enc = encoding_name(entry.encoding);
        println!(
            "  0x{:02x} {:<24} encoding={} offset={} size={} checksum={}",
            entry.section_id,
            name,
            enc,
            entry.offset,
            entry.size,
            hex::encode(entry.checksum),
        );
    }
    println!(
        "Manifest:\n{}",
        serde_json::to_string_pretty(&view.manifest)?
    );
    println!("File hash:    {}", view.file_hash_hex());
    println!("Content hash: {}", view.content_hash_hex()?);
    Ok(())
}

fn run_json(file: PathBuf) -> Result<()> {
    // Reuse the runtime's inspect_json; same schema as
    // `UrnaFile.inspect()` from Python and the PyO3 bindings.
    let rt = urna_runtime::MmapUrnaFile::open(&file)?;
    let s = rt.inspect_json()?;
    // Pretty-print so humans can read it too without a separate tool.
    let v: serde_json::Value = serde_json::from_str(&s)?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}
