//! A C program against the header and the static library: open a corpus
//! built here from `demo/starter/docs.jsonl` with the potion table, ask it, read the hits, take an image, read the
//! schema, close, and the error path on a file that is not there. Runs
//! under `leaks` on macOS when it is available. Skipped, with a note,
//! without a C compiler.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

const PROGRAM: &str = r#"
#include <stdio.h>
#include <string.h>
#include "urna.h"

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    UrnaCorpus *missing = urna_open("/nonexistent/corpus.urna", NULL);
    if (missing != NULL) { printf("opened a missing file\n"); return 1; }
    if (strlen(urna_last_error()) == 0) { printf("no error for a missing file\n"); return 1; }
    UrnaCorpus *c = urna_open(argv[1], argv[2]);
    if (c == NULL) { printf("open failed: %s\n", urna_last_error()); return 1; }
    printf("chunks %zu\n", urna_n_chunks(c));
    char *schema = urna_schema_json(c);
    printf("schema %s\n", schema == NULL ? "(null)" : schema);
    urna_string_free(schema);
    char *hash = urna_content_hash(c);
    printf("content_hash %s\n", hash);
    urna_string_free(hash);
    UrnaHits *h = urna_ask(c, argv[3], 3);
    if (h == NULL) { printf("ask failed: %s\n", urna_last_error()); urna_close(c); return 1; }
    size_t n = urna_hits_len(h);
    printf("hits %zu\n", n);
    for (size_t i = 0; i < n; i++) {
        const char *text = urna_hit_text(h, i);
        printf("  %lld %.3f %s | %.40s\n", (long long)urna_hit_chunk(h, i), urna_hit_score(h, i),
               urna_hit_citation(h, i), text == NULL ? "" : text);
    }
    if (urna_hit_chunk(h, n + 5) != -1) { printf("out of range chunk\n"); return 1; }
    if (urna_hit_text(h, n + 5) != NULL) { printf("out of range text\n"); return 1; }
    size_t len = 7;
    const uint8_t *img = urna_image(c, 0, &len);
    printf("image %s %zu\n", img == NULL ? "none" : "bytes", len);
    UrnaHits *bad = urna_ask(c, "zz:1 text", 3);
    if (bad != NULL) { printf("unknown prefix accepted\n"); return 1; }
    printf("bad prefix: %s\n", urna_last_error());
    urna_hits_free(h);
    urna_hits_free(NULL);
    urna_close(c);
    urna_close(NULL);
    printf("ok\n");
    return 0;
}
"#;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The cargo profile this test runs under: the first directory below the
/// target root (the parent of `CARGO_TARGET_TMPDIR`) on the way to the test
/// binary. Read off the path, not guessed from a fixed number of `parent()`
/// calls: older cargo puts the binary in `target/<profile>/deps/`, newer
/// cargo in `target/<profile>/build/<pkg>/<hash>/`, and a custom profile is
/// named after itself.
fn profile() -> String {
    let exe = std::env::current_exe().unwrap();
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).parent().unwrap();
    exe.strip_prefix(root)
        .ok()
        .and_then(|rel| rel.components().next())
        .and_then(|first| first.as_os_str().to_str())
        .unwrap_or_else(|| panic!("{} is not under {}", exe.display(), root.display()))
        .to_owned()
}

/// The `cargo build` flags for `profile`: none for `debug` (the dev and
/// test profiles), `--release`, or `--profile <name>` for a custom one.
fn profile_args(profile: &str) -> Vec<String> {
    match profile {
        "debug" => Vec::new(),
        "release" => vec!["--release".into()],
        name => vec!["--profile".into(), name.into()],
    }
}

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn staticlib() -> PathBuf {
    // built every time, in the profile this test runs under, so the C
    // program links the library of this checkout and never a stale one.
    // cargo names the file it wrote, wherever its layout puts it.
    let out = Command::new(env!("CARGO"))
        .args([
            "build",
            "-p",
            "urna-cbinds",
            "--message-format=json-render-diagnostics",
        ])
        .args(profile_args(&profile()))
        .current_dir(root())
        .output()
        .expect("cargo build");
    assert!(
        out.status.success(),
        "cargo build -p urna-cbinds:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let lib = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|msg| msg["reason"] == "compiler-artifact")
        .flat_map(|msg| msg["filenames"].as_array().cloned().unwrap_or_default())
        .filter_map(|f| f.as_str().map(PathBuf::from))
        .find(|f| f.file_name().is_some_and(|n| n == "liburna_cbinds.a"))
        .expect("cargo build reported no liburna_cbinds.a");
    assert!(lib.is_file(), "{} missing", lib.display());
    lib
}

/// The starter docs, one chunk each, embedded with the potion table and
/// written under its model name and hash: the corpus the ABI asks.
fn build_corpus(table: &Path, out: &Path) {
    use urna_format::ChunkInput;
    use urna_format::manifest::Manifest;
    use urna_format::writer::UrnaFileBuilder;

    let potion = urna_neural::Potion::open(table).expect("potion table");
    let docs = std::fs::read_to_string(root().join("demo/starter/docs.jsonl")).unwrap();
    let mut chunks = Vec::new();
    let mut at = 0u64;
    for line in docs.lines().filter(|l| !l.trim().is_empty()) {
        let doc: serde_json::Value = serde_json::from_str(line).unwrap();
        let text = doc["text"].as_str().expect("a text field").to_string();
        let end = at + line.len() as u64;
        chunks.push(ChunkInput {
            embedding: potion.embed(&text).unwrap(),
            canonical_text: text,
            source_uri: "docs.jsonl".into(),
            byte_start: at,
            byte_end: end,
        });
        at = end + 1;
    }
    let manifest = Manifest {
        embedding_model: "minishlab/potion-base-8M/v1".into(),
        embedding_dim: u32::try_from(potion.dim()).unwrap(),
        n_chunks: chunks.len() as u64,
        chunker_version: "starter-doc/1".into(),
        model_hash: potion.model_hash().to_string(),
        ..Default::default()
    };
    UrnaFileBuilder::new(manifest)
        .add_chunks(chunks)
        .write_to_path(out)
        .unwrap();
}

#[test]
fn a_c_program_opens_asks_reads_and_closes_without_leaks() {
    if !have("cc") {
        eprintln!("skip: no C compiler");
        return;
    }
    let table = root().join("rust/bridge/python/urna/model/potionb8m");
    let pointer = std::fs::read(table.join("model.safetensors"))
        .is_ok_and(|b| b.starts_with(b"version https://git-lfs"));
    if pointer {
        eprintln!("skip: the potion table is a git-lfs pointer");
        return;
    }
    let dir = std::env::temp_dir().join(format!("urna-cbinds-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("starter.urna");
    build_corpus(&table, &corpus);
    let src = dir.join("main.c");
    std::fs::write(&src, PROGRAM).unwrap();
    let exe = dir.join("main");
    let lib = staticlib();
    let mut cc = Command::new("cc");
    cc.arg("-std=c11")
        .arg("-Wall")
        .arg("-Werror")
        .arg("-I")
        .arg(root().join("rust/cbinds/include"))
        .arg(&src)
        .arg(&lib)
        .arg("-o")
        .arg(&exe);
    if cfg!(target_os = "linux") {
        cc.args(["-lpthread", "-ldl", "-lm"]);
    }
    let out = cc.output().expect("cc runs");
    assert!(
        out.status.success(),
        "cc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = |leaks: bool| {
        let mut cmd = if leaks {
            let mut c = Command::new("leaks");
            c.args(["--atExit", "--"]).arg(&exe);
            c
        } else {
            Command::new(&exe)
        };
        cmd.arg(&corpus).arg(&table).arg("what is urna");
        cmd.output().expect("program runs")
    };
    let out = run(false);
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!("{stdout}");
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("hits 3"), "{stdout}");
    assert!(stdout.contains("urna://"), "{stdout}");
    assert!(
        stdout.contains("bad prefix: filter: unknown search prefix"),
        "{stdout}"
    );
    assert!(stdout.trim_end().ends_with("ok"), "{stdout}");
    if cfg!(target_os = "macos") && Path::new("/usr/bin/leaks").is_file() {
        let out = run(true);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if text.contains("leaks Report Version") {
            assert!(text.contains("0 leaks for 0 total leaked bytes"), "{text}");
        } else {
            eprintln!(
                "note: leaks did not run ({})",
                text.lines().last().unwrap_or("")
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
