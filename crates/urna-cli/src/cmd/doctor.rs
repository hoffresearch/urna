//! `urna doctor` - post-install health check (issue #75).
//!
//! validates the whole install surface a user/agent depends on and exits
//! with a TYPED code so installers, CI, and support can branch on the
//! failure class instead of parsing text:
//!
//!   0  ok (scalar simd still exits 0, printed as a warning)
//!   2  python interpreter missing or not runnable
//!   3  python deps missing (numpy / tokenizers, needed by the embedder)
//!   4  potion embedder script not found (repo layout or data dir)
//!   5  potion model table missing or still a git-lfs pointer
//!   6  embedder run failed (non-zero exit or invalid json contract)
//!
//! checks: urna/format versions, simd backend, python interpreter, python
//! deps, potion embedder presence, potion table presence, and one real
//! offline embed of a fixed probe string. the embedder opens no socket, so
//! doctor itself stays offline-by-construction.

use std::path::Path;
use std::process::Command as ProcCommand;

pub mod codes {
    pub const OK: i32 = 0;
    pub const PYTHON_MISSING: i32 = 2;
    pub const PYTHON_DEPS_MISSING: i32 = 3;
    pub const EMBEDDER_MISSING: i32 = 4;
    pub const POTION_TABLE_MISSING: i32 = 5;
    pub const EMBEDDER_FAILED: i32 = 6;
}

struct Report {
    first_fail: Option<i32>,
}

impl Report {
    fn new() -> Self {
        Self { first_fail: None }
    }
    fn ok(&mut self, what: &str, detail: &str) {
        println!("  ok       {what}: {detail}");
    }
    fn warn(&mut self, what: &str, detail: &str) {
        println!("  warn     {what}: {detail}");
    }
    fn fail(&mut self, code: i32, what: &str, detail: &str) {
        println!("  fail({code}) {what}: {detail}");
        if self.first_fail.is_none() {
            self.first_fail = Some(code);
        }
    }
}

fn run_capture(cmd: &mut ProcCommand) -> Option<std::process::Output> {
    cmd.output().ok().filter(|o| o.status.success())
}

/// the potion table dir lives next to the embedder script
/// (`<root>/forge/embed_query_potion.py` -> `<root>/forge/models/...`).
/// rejects a git-lfs pointer so a fresh clone without `git lfs pull` fails
/// loudly instead of embedding garbage later.
fn check_potion_table(rep: &mut Report, embedder: &Path) {
    let table = embedder.parent().map(|p| {
        p.join("models")
            .join("potion-base-8M")
            .join("model.safetensors")
    });
    let Some(table) = table else {
        rep.fail(
            codes::POTION_TABLE_MISSING,
            "potion table",
            "embedder has no parent dir",
        );
        return;
    };
    match std::fs::read(&table) {
        Ok(bytes) if bytes.starts_with(b"version https://git-lfs") => rep.fail(
            codes::POTION_TABLE_MISSING,
            "potion table",
            &format!(
                "{} is a git-lfs pointer; run `git lfs pull`",
                table.display()
            ),
        ),
        Ok(bytes) => rep.ok(
            "potion table",
            &format!("{} ({:.1} MB)", table.display(), bytes.len() as f64 / 1e6),
        ),
        Err(_) => rep.fail(
            codes::POTION_TABLE_MISSING,
            "potion table",
            &format!("not found: {}", table.display()),
        ),
    }
}

/// one real offline embed: the fixed probe exercises numpy + tokenizers +
/// the table load exactly the way `ask`/`retrieve` do.
fn check_embedder_run(rep: &mut Report, interpreter: &str, embedder: &Path) {
    let out = ProcCommand::new(interpreter)
        .arg(embedder)
        .arg("potion-base-8M")
        .arg("urna doctor probe")
        .output();
    let payload = out.ok().and_then(|o| {
        if o.status.success() {
            serde_json::from_slice::<serde_json::Value>(&o.stdout).ok()
        } else {
            eprintln!("  embedder stderr: {}", String::from_utf8_lossy(&o.stderr));
            None
        }
    });
    match payload {
        Some(v) => {
            let dim = v["embedding_dim"].as_u64().unwrap_or(0) as usize;
            let vec_len = v["vector"].as_array().map(|a| a.len()).unwrap_or(0);
            let hash_ok = v["model_hash"]
                .as_str()
                .map(|h| h.starts_with("sha256:"))
                .unwrap_or(false);
            if dim > 0 && dim == vec_len && hash_ok {
                rep.ok(
                    "embedder run",
                    &format!(
                        "dim={dim} model_hash={}",
                        v["model_hash"].as_str().unwrap_or("")
                    ),
                );
            } else {
                rep.fail(
                    codes::EMBEDDER_FAILED,
                    "embedder run",
                    "json contract broken (dim/vector/model_hash)",
                );
            }
        }
        None => rep.fail(
            codes::EMBEDDER_FAILED,
            "embedder run",
            "embedder exited non-zero or emitted invalid json",
        ),
    }
}

pub fn run() -> anyhow::Result<()> {
    let mut rep = Report::new();
    println!("urna doctor");

    rep.ok(
        "version",
        &format!(
            "urna {}, format v{}",
            env!("CARGO_PKG_VERSION"),
            urna_format::layout::URNA_FORMAT_VERSION
        ),
    );

    let simd = urna_runtime::simd::detect_backend();
    if simd == urna_runtime::SimdBackend::Scalar {
        rep.warn(
            "simd",
            "scalar fallback (unset URNA_FORCE_SCALAR, or the cpu lacks avx2/neon)",
        );
    } else {
        rep.ok("simd", simd.name());
    }

    let interpreter = super::pyenv::resolve_interpreter();
    let py_ok = run_capture(ProcCommand::new(&interpreter).arg("--version"))
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    match py_ok {
        Some(ver) => rep.ok("python", &format!("{interpreter} ({ver})")),
        None => {
            rep.fail(
                codes::PYTHON_MISSING,
                "python",
                &format!("not runnable: {interpreter} (set URNA_PYTHON)"),
            );
            // deps and embedder both need python; report and exit.
            println!("urna doctor: failed");
            std::process::exit(rep.first_fail.unwrap_or(codes::PYTHON_MISSING));
        }
    }

    let deps_ok = run_capture(
        ProcCommand::new(&interpreter)
            .arg("-c")
            .arg("import numpy, tokenizers"),
    )
    .is_some();
    if deps_ok {
        rep.ok("python deps", "numpy, tokenizers");
    } else {
        rep.fail(
            codes::PYTHON_DEPS_MISSING,
            "python deps",
            "numpy and/or tokenizers missing (uv pip install numpy tokenizers)",
        );
    }

    let embedder = super::embed_gate::default_potion_embedder_path();
    if !embedder.exists() {
        rep.fail(
            codes::EMBEDDER_MISSING,
            "embedder",
            &format!("not found: {}", embedder.display()),
        );
        println!("urna doctor: failed");
        std::process::exit(rep.first_fail.unwrap_or(codes::EMBEDDER_MISSING));
    }
    rep.ok("embedder", &format!("{}", embedder.display()));

    check_potion_table(&mut rep, &embedder);
    if deps_ok {
        check_embedder_run(&mut rep, &interpreter, &embedder);
    } else {
        rep.warn("embedder run", "skipped (python deps missing)");
    }

    match rep.first_fail {
        None => {
            println!("urna doctor: ok");
            std::process::exit(codes::OK);
        }
        Some(code) => {
            println!("urna doctor: failed");
            std::process::exit(code);
        }
    }
}
