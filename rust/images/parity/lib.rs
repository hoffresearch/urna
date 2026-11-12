//! The crate exists for `tests/`, which compare urna-images and
//! urna-av1dec with the decoders from crates.io, kept out of the urna
//! workspace. This file only gathers what they compare on.

#![allow(
    clippy::expect_used,
    reason = "a test helper: a failure is the finding"
)]

use std::path::Path;

/// The AVIF files to compare, as `(label, bytes)`: every `.avif` of the
/// directory named by `URNA_AVIF_DIR` (what CI encodes, see
/// `.github/workflows/gatecheck.yml`), else every inlined AVIF blob of the
/// corpus named by `URNA_CARDS_CORPUS`, at most `URNA_PARITY_LIMIT` of
/// them. `None` when neither is set: the test says so and passes.
pub fn avif_inputs() -> Option<Vec<(String, Vec<u8>)>> {
    let limit: usize = std::env::var("URNA_PARITY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    if let Some(dir) = std::env::var_os("URNA_AVIF_DIR") {
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .expect("URNA_AVIF_DIR lists")
            .map(|e| e.expect("URNA_AVIF_DIR entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "avif"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "URNA_AVIF_DIR holds no .avif file");
        return Some(
            files
                .into_iter()
                .take(limit)
                .map(|p| {
                    let bytes = std::fs::read(&p).expect("avif reads");
                    (p.display().to_string(), bytes)
                })
                .collect(),
        );
    }
    let path = std::env::var_os("URNA_CARDS_CORPUS")?;
    let rt = urna_engine::MmapUrnaFile::open(Path::new(&path)).expect("corpus opens");
    let refs = rt.blob_refs().expect("the corpus has blobs");
    Some(
        refs.iter()
            .enumerate()
            .filter(|(_, r)| r.inlined && r.original_uri.ends_with(".avif"))
            .take(limit)
            .map(|(i, _)| {
                let bytes = rt.blob_bytes(i).expect("blob bytes").to_vec();
                (format!("blob {i}"), bytes)
            })
            .collect(),
    )
}
