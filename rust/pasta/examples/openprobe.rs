//! Where the open time goes: the format's two verification passes, then the
//! two hashes the runtime recomputes, then the whole runtime open.
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("FILE.urna");
    let f = std::fs::File::open(&path).expect("open");
    // SAFETY: read-only map of a file nobody writes while the probe runs.
    let mmap = unsafe { memmap2::Mmap::map(&f) }.expect("mmap");
    let t = Instant::now();
    let _ = mmap.iter().step_by(4096).fold(0u8, |a, b| a ^ b);
    println!("page in        {:>7.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let view = urna_format::reader::UrnaView::from_bytes(&mmap).expect("view");
    println!(
        "from_bytes     {:>7.0} ms  (section checksums + footer hash)",
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    let _ = view.file_hash_hex();
    println!("file_hash      {:>7.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let _ = view.content_hash_hex().expect("content");
    println!("content_hash   {:>7.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let _ = urna_engine::MmapUrnaFile::open(std::path::Path::new(&path)).expect("runtime");
    println!(
        "runtime open   {:>7.0} ms  (everything, warm cache)",
        t.elapsed().as_secs_f64() * 1e3
    );
}
