#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use std::process::Command;

use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

fn build_test_file(path: &PathBuf, dim: usize, n: usize) {
    let mut builder = UrnaFileBuilder::new(Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    });
    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("text_{}", i),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: emb,
        });
    }
    builder.write_to_path(path).unwrap();
}

#[test]
fn cli_validate_ok() {
    let path = tmp_path("cli_validate.urna");
    let _ = std::fs::remove_file(&path);
    build_test_file(&path, 8, 10);

    let bin = env!("CARGO_BIN_EXE_urna");
    let out = Command::new(bin)
        .args(["validate", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("valid .urna v1 file"));
    assert!(stdout.contains("Required sections:"));
    assert!(stdout.contains("File hash:"));
    assert!(stdout.contains("Content hash:"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_search_ok() {
    let path = tmp_path("cli_search.urna");
    let _ = std::fs::remove_file(&path);
    build_test_file(&path, 4, 5);

    let bin = env!("CARGO_BIN_EXE_urna");

    // stats
    let out = Command::new(bin)
        .args(["stats", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("chunks:       5"));
    assert!(stdout.contains("dtype:        float32"));
    assert!(stdout.contains("metric:       ip"));

    // search aligned to first axis returns chunk that maps to axis 0
    let query = "[1.0, 0.0, 0.0, 0.0]";
    let out = Command::new(bin)
        .args(["search", path.to_str().unwrap(), query, "-k", "1"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("index_type:"));
    assert!(stdout.contains("recall:"));
    assert!(stdout.contains("score=1.000000"));
    assert!(stdout.contains("citation_id=urna://sha256:"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_cite_resolves_citation() {
    let path = tmp_path("cli_cite.urna");
    let _ = std::fs::remove_file(&path);
    build_test_file(&path, 4, 2);

    let bin = env!("CARGO_BIN_EXE_urna");

    // First, fetch a real citation_id by running search.
    let q = "[1.0, 0.0, 0.0, 0.0]";
    let out = Command::new(bin)
        .args(["search", path.to_str().unwrap(), q, "-k", "1"])
        .output()
        .unwrap();
    assert!(out.status.success(), "search failed");
    let stdout = String::from_utf8_lossy(&out.stdout);

    let cit_token = stdout
        .split_whitespace()
        .find(|t| t.starts_with("citation_id=urna://"))
        .expect("citation_id present in search output");
    let citation = cit_token
        .strip_prefix("citation_id=")
        .expect("citation_id= prefix");

    let out = Command::new(bin)
        .args(["cite", path.to_str().unwrap(), citation])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cite failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("source_uri:   doc.txt"));
    assert!(stdout.contains("byte_start:"));
    assert!(stdout.contains("byte_end:"));
    assert!(stdout.contains("text_"));

    // Mismatched content_hash → cite must fail loudly.
    let bogus = format!("urna://sha256:{}/sha256:{}", "0".repeat(64), "0".repeat(64));
    let out = Command::new(bin)
        .args(["cite", path.to_str().unwrap(), &bogus])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "cite should reject content_hash mismatch"
    );

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// flagship verbs: ask + retrieve. these embed the query OFFLINE with the
// default potion static table, so they need a python carrying numpy +
// tokenizers + the vendored potion table (git-lfs). when that toolchain is
// absent (a minimal CI runner), the helpers below return None and the test
// skips with a printed note rather than failing - the pure-rust suite above
// is unconditional. set URNA_PYTHON to point at the venv that has the deps.
// ---------------------------------------------------------------------------

/// resolve a python interpreter that can import forge.embed_potion (numpy +
/// tokenizers + the vendored table). prefers $URNA_PYTHON, then .venv at the
/// repo root, then `python3`. returns None when none can build the demo.
fn forge_python() -> Option<(String, PathBuf)> {
    // repo root: this test file is crates/urna-cli/tests/, go up three.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)?;
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("URNA_PYTHON") {
        candidates.push(p);
    }
    candidates.push(root.join(".venv/bin/python").to_string_lossy().into_owned());
    candidates.push("python3".into());
    for py in candidates {
        let probe = Command::new(&py)
            .arg("-c")
            .arg("import numpy, tokenizers")
            .current_dir(&root)
            .output();
        if matches!(probe, Ok(o) if o.status.success()) {
            return Some((py, root));
        }
    }
    None
}

/// build the cc0 demo corpus into `path` via forge.retrieve.build_demo with
/// the offline potion embedder. returns false (skip) when forge deps absent.
fn build_demo_corpus(py: &str, root: &std::path::Path, path: &std::path::Path) -> bool {
    let code = format!(
        "import sys; sys.path.insert(0, 'python'); \
         from forge.retrieve import build_demo; build_demo({:?})",
        path.to_string_lossy()
    );
    let out = Command::new(py)
        .args(["-c", &code])
        .current_dir(root)
        .output()
        .expect("spawn python build_demo");
    if !out.status.success() {
        eprintln!(
            "build_demo failed (skipping): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return false;
    }
    path.exists()
}

#[test]
fn cli_ask_answer_is_cited_text_only_and_explain_adds_honesty_line() {
    let Some((py, root)) = forge_python() else {
        eprintln!("skip cli_ask: no python with numpy+tokenizers+potion table");
        return;
    };
    let path = tmp_path("cli_ask_demo.urna");
    let _ = std::fs::remove_file(&path);
    if !build_demo_corpus(&py, &root, &path) {
        eprintln!("skip cli_ask: demo corpus build skipped");
        return;
    }

    let bin = env!("CARGO_BIN_EXE_urna");
    let query = "can I use this offline with no network";

    // --disclose answer (default): cited text + a urna:// citation, no
    // field-wall (no "score=", no "rerank_source", no "route:").
    let out = Command::new(bin)
        .env("URNA_PYTHON", &py)
        .current_dir(&root)
        .args(["ask", path.to_str().unwrap(), query, "-k", "1"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "ask answer failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let answer = String::from_utf8_lossy(&out.stdout);
    assert!(
        answer.contains("urna://sha256:"),
        "answer must cite a urna:// uri"
    );
    assert!(
        answer.contains("offline"),
        "answer must carry the cited text"
    );
    assert!(
        !answer.contains("rerank_source"),
        "answer level must NOT print the explain honesty line"
    );
    assert!(
        !answer.contains("route:"),
        "answer level must NOT print the route/field-wall"
    );

    // --disclose explain: ALSO prints the rerank-source honesty line; the f32
    // demo corpus is full precision, so it says exactly "real cosine".
    let out = Command::new(bin)
        .env("URNA_PYTHON", &py)
        .current_dir(&root)
        .args([
            "ask",
            path.to_str().unwrap(),
            query,
            "-k",
            "1",
            "--disclose",
            "explain",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let explain = String::from_utf8_lossy(&out.stdout);
    assert!(explain.contains("rerank_source: real cosine"));
    assert!(explain.contains("route:         exact"));
    assert!(explain.contains("urna://sha256:"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_retrieve_answer_pack_score_equals_search_and_cite_round_trips() {
    let Some((py, root)) = forge_python() else {
        eprintln!("skip cli_retrieve: no python with forge deps");
        return;
    };
    let path = tmp_path("cli_retrieve_demo.urna");
    let _ = std::fs::remove_file(&path);
    if !build_demo_corpus(&py, &root, &path) {
        eprintln!("skip cli_retrieve: demo corpus build skipped");
        return;
    }

    let bin = env!("CARGO_BIN_EXE_urna");
    let query = "how do citations prove a source";

    // jsonl answer-pack: every line is a valid object with the answer-pack
    // fields and score_type=cosine.
    let out = Command::new(bin)
        .env("URNA_PYTHON", &py)
        .current_dir(&root)
        .args(["retrieve", path.to_str().unwrap(), query, "-k", "3"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "retrieve failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let jsonl = String::from_utf8_lossy(&out.stdout);
    let mut citation = String::new();
    let mut text = String::new();
    let mut count = 0;
    for line in jsonl.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("each jsonl line is valid json");
        assert_eq!(v["score_type"], "cosine");
        assert!(v["score"].is_number());
        let cit = v["citation_id"].as_str().unwrap();
        assert!(cit.starts_with("urna://sha256:"), "well-formed citation");
        // citation is urna://<content_hash>/<chunk_id>, both sha256:.
        let body = cit.strip_prefix("urna://").unwrap();
        let (ch, ci) = body.split_once('/').unwrap();
        assert_eq!(ch, v["content_hash"].as_str().unwrap());
        assert_eq!(ci, v["chunk_id"].as_str().unwrap());
        if count == 0 {
            citation = cit.to_string();
            text = v["text"].as_str().unwrap().to_string();
        }
        count += 1;
    }
    assert!(count > 0, "retrieve emitted at least one hit");

    // cite resolves that exact citation, and its text == the answer-pack text
    // (tier-1 stored canonical, never an original-byte reopen).
    let out = Command::new(bin)
        .current_dir(&root)
        .args(["cite", path.to_str().unwrap(), &citation])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cite round-trip failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let cited = String::from_utf8_lossy(&out.stdout);
    // the canonical text body from cite must contain the retrieve text (cite
    // prints a header then the text; the stored bytes are identical).
    let trimmed = text.trim();
    let snippet: String = trimmed.lines().next().unwrap_or("").into();
    assert!(
        cited.contains(snippet.trim()),
        "cite text must equal the retrieve answer-pack stored canonical text"
    );

    // a content_hash-mismatched cite still fails loudly (existing behavior).
    let bogus = format!("urna://sha256:{}/sha256:{}", "0".repeat(64), "0".repeat(64));
    let out = Command::new(bin)
        .current_dir(&root)
        .args(["cite", path.to_str().unwrap(), &bogus])
        .output()
        .unwrap();
    assert!(!out.status.success(), "mismatched cite must fail");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_inspect_shows_section_names() {
    let path = tmp_path("cli_inspect.urna");
    let _ = std::fs::remove_file(&path);
    build_test_file(&path, 4, 2);

    let bin = env!("CARGO_BIN_EXE_urna");
    let out = Command::new(bin)
        .args(["inspect", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("chunk_ids"));
    assert!(stdout.contains("chunks_canonical"));
    assert!(stdout.contains("chunks_original_spans"));
    assert!(stdout.contains("embeddings"));
    assert!(stdout.contains("provenance"));
    assert!(stdout.contains("search_contract"));

    let _ = std::fs::remove_file(&path);
}
