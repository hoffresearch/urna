//! `urna media <file> [--export DIR]` — list the media blobs a corpus
//! references, and export the inlined ones (0x17) back to standalone
//! files. Export proves each blob against its blob_refs content_hash
//! BEFORE writing, so a corrupt section can never fan out to disk.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use urna_runtime::MmapUrnaFile;

pub fn run(file: PathBuf, export: Option<PathBuf>) -> Result<()> {
    let urna = MmapUrnaFile::open(&file)?;
    let Some(refs) = urna.blob_refs() else {
        println!("no media: {} declares no blob_refs (0x14)", file.display());
        return Ok(());
    };
    let refs = refs.to_vec();
    let inlined_total: u64 = refs.iter().filter(|r| r.inlined).map(|r| r.byte_len).sum();
    println!(
        "media blobs: {} ({} inlined, {} sidecar), inlined bytes: {}",
        refs.len(),
        refs.iter().filter(|r| r.inlined).count(),
        refs.iter().filter(|r| !r.inlined).count(),
        inlined_total,
    );
    for (i, r) in refs.iter().enumerate() {
        println!(
            "  [{}] {} {} bytes {}",
            i,
            r.original_uri,
            r.byte_len,
            if r.inlined { "inlined" } else { "sidecar" },
        );
    }
    let Some(dir) = export else { return Ok(()) };
    if !urna.has_blob_data() {
        bail!(
            "{} has no blob_data (0x17): its media lives in the sidecar files listed above",
            file.display()
        );
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create export dir {}", dir.display()))?;
    let mut exported = 0usize;
    for (i, r) in refs.iter().enumerate() {
        if !r.inlined {
            continue;
        }
        let bytes = urna.blob_bytes(i)?;
        let digest = Sha256::digest(bytes);
        if digest[..] != r.content_hash {
            bail!(
                "blob {} ({}) failed its content_hash check; refusing to export",
                i,
                r.original_uri
            );
        }
        // media:// uris are relative names by construction; take the final
        // component so a hostile uri can never escape the export dir.
        let name = r
            .original_uri
            .trim_start_matches("media://")
            .rsplit('/')
            .next()
            .filter(|n| !n.is_empty() && *n != "." && *n != "..")
            .with_context(|| format!("blob {} has no exportable name: {}", i, r.original_uri))?;
        let out = dir.join(name);
        std::fs::write(&out, bytes).with_context(|| format!("write {}", out.display()))?;
        exported += 1;
    }
    println!("exported {} blobs to {}", exported, dir.display());
    Ok(())
}
