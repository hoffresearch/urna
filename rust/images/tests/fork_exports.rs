//! The embedded AV1 decoder (`rust/av1dec`, a fork of rav1d) exports no C
//! name.
//!
//! Upstream marks its `dav1d_*` entry points, its tables and its cpu flags
//! `#[no_mangle]`. With those names exported, the crates.io rav1d that
//! `rust/images/parity` links as the reference resolved to the fork's
//! definitions, one symbol per name, and the plane parity compared the fork
//! with itself; `liburna_cbinds` exported them too. The parity workspace
//! checks one entry point by address (`the_reference_is_not_the_embedded_fork`),
//! and runs only where that workspace does. This reads the fork's source,
//! so a name that comes back on any item, a table as much as an entry
//! point, fails in the workspace's own test run.

#![allow(
    clippy::expect_used,
    reason = "a test helper: a failure is the finding"
)]

use std::path::{Path, PathBuf};

/// The attributes that give an item a C symbol name.
const EXPORTS: [&str; 2] = ["no_mangle", "export_name"];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the fork's directory lists") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `file:line: text` for every line of `files` that names an export.
fn exported(files: &[PathBuf]) -> Vec<String> {
    let mut found = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(file).expect("a source file reads");
        for (n, line) in text.lines().enumerate() {
            if EXPORTS.iter().any(|e| line.contains(e)) {
                found.push(format!("{}:{}: {}", file.display(), n + 1, line.trim()));
            }
        }
    }
    found
}

#[test]
fn the_embedded_decoder_exports_no_c_name() {
    let fork = Path::new(env!("CARGO_MANIFEST_DIR")).join("../av1dec");
    let mut files = Vec::new();
    rust_files(&fork, &mut files);
    // the files upstream exported from: a scan that missed them proves nothing.
    for name in ["src/lib.rs", "src/tables.rs", "src/cpu.rs"] {
        assert!(
            files.iter().any(|f| f.ends_with(name)),
            "{name} is not among the {} files read under {}",
            files.len(),
            fork.display()
        );
    }
    let found = exported(&files);
    assert!(
        found.is_empty(),
        "rust/av1dec exports a C name again (see its UPSTREAM, \"no #[no_mangle]\"):\n{}",
        found.join("\n")
    );
}

#[test]
fn an_export_in_a_source_file_is_found() {
    let dir = std::env::temp_dir().join(format!("urna-fork-exports-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let file = dir.join("tables.rs");
    std::fs::write(
        &file,
        "pub static A: [u8; 1] = [0];\n#[no_mangle]\npub static dav1d_b: [u8; 1] = [0];\n\
         #[unsafe(export_name = \"dav1d_c\")]\npub fn c() {}\n",
    )
    .expect("the scratch file writes");
    let mut files = Vec::new();
    rust_files(&dir, &mut files);
    let found = exported(&files);
    std::fs::remove_dir_all(&dir).expect("the scratch directory goes");
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found[0].ends_with(":2: #[no_mangle]"), "{found:?}");
    assert!(
        found[1].contains(":4: ") && found[1].contains("dav1d_c"),
        "{found:?}"
    );
}
