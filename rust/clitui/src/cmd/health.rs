//! The install health checks as data: one ordered list of rows that
//! `urna doctor` prints and the terminal ui draws. The order, the stop
//! points and the typed codes are the `doctor` contract (see its module
//! doc); nothing here prints.

use std::path::Path;
use std::process::Command as ProcCommand;

use super::doctor::codes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Ok,
    Warn,
    Fail(i32),
}

#[derive(Clone, Debug)]
pub struct Row {
    pub level: Level,
    pub what: &'static str,
    pub detail: String,
    /// the embedder's stderr when its probe run failed.
    pub stderr: Option<String>,
}

impl Row {
    fn new(level: Level, what: &'static str, detail: impl Into<String>) -> Self {
        Self {
            level,
            what,
            detail: detail.into(),
            stderr: None,
        }
    }
}

/// The exit code of a finished check list: the first failure, else 0.
pub fn exit_code(rows: &[Row]) -> i32 {
    rows.iter()
        .find_map(|r| match r.level {
            Level::Fail(c) => Some(c),
            _ => None,
        })
        .unwrap_or(codes::OK)
}

fn run_capture(cmd: &mut ProcCommand) -> Option<std::process::Output> {
    cmd.output().ok().filter(|o| o.status.success())
}

/// Every check, in order. A missing embedder stops the list there (the
/// later checks need it). The embedder is the one `ask` resolves from here
/// (`embed_gate`'s ladder: the checkout first).
pub fn collect() -> Vec<Row> {
    collect_with(None, false)
}

/// The checks plus the corpus-building toolchain (python with numpy and
/// tokenizers, the embedder script under it): `urna doctor --build`.
pub fn collect_build() -> Vec<Row> {
    collect_with(None, true)
}

/// The same checks on the payload installed in `home` (`<root>/urna`), the
/// one setup lays down, whatever the current directory resolves to: a
/// checkout nearby must not pass setup's verify for a payload that cannot
/// serve `ask`.
#[cfg(feature = "tui")]
pub fn collect_installed(home: &Path) -> Vec<Row> {
    collect_with(Some(home), false)
}

/// The query surface first: the payload, the potion table and one real
/// embed in process, which is what `ask` needs and all it needs. Python
/// comes after as the corpus-building toolchain: a warning when it is
/// missing, a failure with codes 2 and 3 only when `build` asks for it.
fn collect_with(installed: Option<&Path>, build: bool) -> Vec<Row> {
    let mut rows = Vec::new();
    rows.push(Row::new(
        Level::Ok,
        "version",
        format!(
            "urna {}, format v{}",
            super::buildid::NAME,
            urna_format::layout::URNA_FORMAT_VERSION
        ),
    ));

    let simd = urna_engine::simd::detect_backend();
    if simd == urna_engine::SimdBackend::Scalar {
        rows.push(Row::new(
            Level::Warn,
            "simd",
            "scalar fallback (unset URNA_FORCE_SCALAR, or the cpu lacks avx2/neon)",
        ));
    } else {
        rows.push(Row::new(Level::Ok, "simd", simd.name()));
    }

    if let Some(home) = installed {
        let missing = super::payload::missing(home);
        rows.push(match missing.as_slice() {
            [] => Row::new(Level::Ok, "payload", home.display().to_string()),
            gone => Row::new(
                Level::Fail(codes::EMBEDDER_MISSING),
                "payload",
                format!(
                    "{} is missing {} (run `urna setup`)",
                    home.display(),
                    gone.join(", ")
                ),
            ),
        });
    }

    let embedder = match installed {
        Some(home) => ["python", "urna", "embed", "potionqry.py"]
            .iter()
            .fold(home.to_path_buf(), |p, c| p.join(c)),
        None => super::embed_gate::default_potion_embedder_path(),
    };
    if !embedder.exists() {
        rows.push(Row::new(
            Level::Fail(codes::EMBEDDER_MISSING),
            "embedder",
            match super::paths::data_roots()
                .first()
                .filter(|_| installed.is_none())
            {
                Some(root) => format!(
                    "not found (looked in the repo layout and {}) (run `urna setup`)",
                    super::payload::package_in(root).display()
                ),
                None => format!("not found: {} (run `urna setup`)", embedder.display()),
            },
        ));
        return rows;
    }
    rows.push(Row::new(
        Level::Ok,
        "embedder",
        embedder.display().to_string(),
    ));

    let table = potion_table(&embedder);
    let table_ok = table.level == Level::Ok;
    rows.push(table);
    if table_ok {
        rows.push(potion_run(&embedder));
    }

    // the corpus-building toolchain. `ask` never needs it, so the rows warn
    // unless the doctor was asked about building.
    let level = |code: i32| {
        if build {
            Level::Fail(code)
        } else {
            Level::Warn
        }
    };
    let was_quiet = super::pyenv::is_quiet();
    super::pyenv::set_quiet(true);
    let interpreter = super::pyenv::resolve_interpreter();
    super::pyenv::set_quiet(was_quiet);
    if !super::pyenv::may_run(&interpreter) {
        rows.push(Row::new(
            level(codes::PYTHON_MISSING),
            "python (build)",
            super::pyenv::stub_hint(&interpreter),
        ));
        return rows;
    }
    let py_ver = run_capture(ProcCommand::new(&interpreter).arg("--version"))
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    match py_ver {
        Some(ver) => rows.push(Row::new(
            Level::Ok,
            "python (build)",
            format!("{interpreter} ({ver})"),
        )),
        None => {
            rows.push(Row::new(
                level(codes::PYTHON_MISSING),
                "python (build)",
                format!(
                    "not runnable: {interpreter}; only corpus builds need it (set URNA_PYTHON, or run `urna setup --python`)"
                ),
            ));
            return rows;
        }
    }

    let deps_ok = run_capture(
        ProcCommand::new(&interpreter)
            .arg("-c")
            .arg("import numpy, tokenizers"),
    )
    .is_some();
    if deps_ok {
        rows.push(Row::new(
            Level::Ok,
            "python deps (build)",
            "numpy, tokenizers",
        ));
    } else {
        rows.push(Row::new(
            level(codes::PYTHON_DEPS_MISSING),
            "python deps (build)",
            "numpy and/or tokenizers missing; only corpus builds need them (run `urna setup --python`, or uv pip install numpy tokenizers)",
        ));
    }
    if build {
        if deps_ok {
            rows.push(embedder_run(&interpreter, &embedder));
        } else {
            rows.push(Row::new(
                Level::Warn,
                "embedder run (build)",
                "skipped (python deps missing)",
            ));
        }
    }
    rows
}

/// one real embed in process: the potion table through urna's own runtime,
/// exactly the way `ask`, `retrieve` and the ask tab embed a query.
fn potion_run(embedder: &Path) -> Row {
    let fail = |d: String| Row::new(Level::Fail(codes::EMBEDDER_FAILED), "embed run", d);
    let Some(dir) = super::potion::table_dir(embedder) else {
        return fail("the potion table dir is not beside the embedder script".into());
    };
    let potion = match super::potion::shared(&dir) {
        Ok(p) => p,
        Err(e) => return fail(format!("{e:#}")),
    };
    match potion.embed("urna doctor probe") {
        Ok(v) if v.len() == potion.dim() && potion.model_hash().starts_with("sha256:") => Row::new(
            Level::Ok,
            "embed run",
            format!(
                "in process, dim={} model_hash={}",
                potion.dim(),
                potion.model_hash()
            ),
        ),
        Ok(v) => fail(format!("vector of {} for dim {}", v.len(), potion.dim())),
        Err(e) => fail(format!("{e:#}")),
    }
}

/// the potion table dir lives in the same package as the embedder script
/// (`<pkg>/embed/potionqry.py` -> `<pkg>/model/potionb8m/...`).
/// rejects a git-lfs pointer so a fresh clone without `git lfs pull` fails
/// loudly instead of embedding garbage later.
fn potion_table(embedder: &Path) -> Row {
    let fail = |d: String| Row::new(Level::Fail(codes::POTION_TABLE_MISSING), "potion table", d);
    let Some(table) = embedder
        .parent()
        .and_then(Path::parent)
        .map(|p| p.join("model").join("potionb8m").join("model.safetensors"))
    else {
        return fail("embedder has no parent dir".into());
    };
    match std::fs::read(&table) {
        Ok(bytes) if bytes.starts_with(b"version https://git-lfs") => fail(format!(
            "{} is a git-lfs pointer; run `git lfs pull`",
            table.display()
        )),
        Ok(bytes) => Row::new(
            Level::Ok,
            "potion table",
            format!("{} ({:.1} MB)", table.display(), bytes.len() as f64 / 1e6),
        ),
        Err(_) => fail(format!("not found: {} (run `urna setup`)", table.display())),
    }
}

/// one real embed through the python script, the corpus-building path:
/// the fixed probe exercises numpy + tokenizers + the table under the
/// interpreter a build would use.
fn embedder_run(interpreter: &str, embedder: &Path) -> Row {
    let out = ProcCommand::new(interpreter)
        .arg(embedder)
        .arg("potion-base-8M")
        .arg("urna doctor probe")
        .output();
    let mut stderr = None;
    let payload = out.ok().and_then(|o| {
        if o.status.success() {
            serde_json::from_slice::<serde_json::Value>(&o.stdout).ok()
        } else {
            stderr = Some(String::from_utf8_lossy(&o.stderr).into_owned());
            None
        }
    });
    let fail = |d: &str| {
        Row::new(
            Level::Fail(codes::EMBEDDER_FAILED),
            "embedder run (build)",
            d,
        )
    };
    let Some(v) = payload else {
        let mut row = fail("embedder exited non-zero or emitted invalid json");
        row.stderr = stderr;
        return row;
    };
    let dim = v["embedding_dim"].as_u64().unwrap_or(0) as usize;
    let vec_len = v["vector"].as_array().map(|a| a.len()).unwrap_or(0);
    let hash = v["model_hash"].as_str().unwrap_or("");
    if dim > 0 && dim == vec_len && hash.starts_with("sha256:") {
        Row::new(
            Level::Ok,
            "embedder run (build)",
            format!("dim={dim} model_hash={hash}"),
        )
    } else {
        fail("json contract broken (dim/vector/model_hash)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "tui")]
    #[test]
    fn verify_fails_a_payload_missing_a_file_wherever_it_runs() {
        // the checkout this test runs in holds every script; the installed
        // payload is what counts.
        let home = std::env::temp_dir().join(format!("urna_health_{}", std::process::id()));
        std::fs::create_dir_all(home.join("python/urna")).unwrap();
        let rows = collect_installed(&home);
        let row = rows.iter().find(|r| r.what == "payload").unwrap();
        assert_eq!(row.level, Level::Fail(codes::EMBEDDER_MISSING));
        assert!(row.detail.contains("potionqry.py"), "{}", row.detail);
        assert_ne!(exit_code(&rows), 0);
        std::fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn exit_code_is_the_first_failure() {
        let rows = vec![
            Row::new(Level::Ok, "a", ""),
            Row::new(Level::Warn, "b", ""),
            Row::new(Level::Fail(4), "c", ""),
            Row::new(Level::Fail(6), "d", ""),
        ];
        assert_eq!(exit_code(&rows), 4);
    }

    #[test]
    fn exit_code_ignores_warnings() {
        let rows = vec![Row::new(Level::Warn, "simd", "scalar")];
        assert_eq!(exit_code(&rows), codes::OK);
    }

    #[test]
    fn collect_always_reports_the_version_first() {
        let rows = collect();
        assert_eq!(rows[0].what, "version");
        assert_eq!(rows[0].level, Level::Ok);
    }
}
