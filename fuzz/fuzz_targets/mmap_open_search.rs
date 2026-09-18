#![no_main]
//! The whole runtime path a hostile file reaches: bytes on disk, `mmap`,
//! `MmapUrnaFile::open` (every index / blob / space codec), then every
//! search verb. Slower than the in-memory targets (one temp file per
//! input); keep `-rss_limit_mb` generous, the open path is allowed to
//! allocate up to the capped decompression sizes.

#[path = "reseal.rs"]
mod reseal;

use libfuzzer_sys::fuzz_target;
use urna_runtime::MmapUrnaFile;

fuzz_target!(|data: &[u8]| {
    let Some(bytes) = reseal::split(data) else {
        return;
    };
    let path = std::env::temp_dir().join(format!("urna_fuzz_{}.urna", std::process::id()));
    if std::fs::write(&path, &bytes).is_err() {
        return;
    }
    if let Ok(f) = MmapUrnaFile::open(&path) {
        let dim = f.embedding_dim();
        let q: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.11).sin()).collect();
        let _ = f.search(&q, 5);
        let _ = f.search_ann(&q, 5, 32);
        let _ = f.search_hybrid(&q, "alpha term3 shared2", 5, 16);
        let _ = f.search_graph(&q, 5, 2, 32);
        for name in f.space_names().iter().map(|s| s.to_string()) {
            let _ = f.search_space(&name, &[0.6, 0.8], 5, None);
        }
        let _ = f.blob_bytes(0);
        let _ = f.inspect_json();
        let _ = f.revalidate();
    }
    let _ = std::fs::remove_file(&path);
});
