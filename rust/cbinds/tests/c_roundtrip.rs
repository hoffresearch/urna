//! A C program against the header and the static library: open the
//! quickstart corpus, ask it, read the hits, take an image, read the
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

/// `target/<profile>/` of this build, from the test binary's own path.
fn target_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent().unwrap().parent().unwrap().to_path_buf()
}

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn staticlib() -> PathBuf {
    let lib = target_dir().join("liburna_cbinds.a");
    if !lib.is_file() {
        let status = Command::new(env!("CARGO"))
            .args(["build", "-p", "urna-cbinds"])
            .current_dir(root())
            .status()
            .expect("cargo build");
        assert!(status.success(), "cargo build -p urna-cbinds");
    }
    assert!(lib.is_file(), "{} missing", lib.display());
    lib
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
    let corpus = root().join("demo/starter/out/quickstart.urna");
    assert!(corpus.is_file(), "{} missing", corpus.display());
    let dir = std::env::temp_dir().join(format!("urna-cbinds-c-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
