//! `ask`, `retrieve` and `search-text` find their python embedders outside a
//! checkout: the payload lays the `urna` package down at
//! `<data root>/urna/python/urna/`, `searchtxt.py` and the registry embedder
//! in its `embed/`, and the one resolution ladder
//! (`embed_gate::installed_script_in`) reaches both there.
//! the binary is copied out of `target/` first, because a dev-built binary
//! also resolves against its own checkout and would always find the repo's
//! copy; the cwd is a temp dir, never the repo.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("urna_resolve_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The binary copied to `<dir>/bin/`, so no checkout sits above it.
fn detached_binary(dir: &Path) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_BIN_EXE_urna"));
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let dst = bin.join(src.file_name().unwrap());
    std::fs::copy(&src, &dst).unwrap();
    dst
}

/// `<verb> <golden fixture> q` from `cwd`, every data root pointed into
/// `dir`; the interpreter is a path that does not exist, so the run stops
/// right after the embedder is resolved and names it. returns stderr.
fn run(bin: &Path, dir: &Path, cwd: &Path, verb: &str) -> String {
    let fixture = repo().join("rust/format/tests/fixtures/golden_v1_minimal.urna");
    let out = output(
        Command::new(bin)
            .args([verb, fixture.to_str().unwrap(), "q"])
            .current_dir(cwd)
            .env("URNA_DATA_DIR", dir.join("data"))
            .env("XDG_DATA_HOME", dir.join("xdg"))
            .env("HOME", dir.join("home"))
            .env_remove("LOCALAPPDATA")
            .env("URNA_PYTHON", dir.join("no-such-python")),
    );
    assert!(!out.status.success());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The checkout's `urna` package.
fn package() -> PathBuf {
    repo()
        .join("rust")
        .join("bridge")
        .join("python")
        .join("urna")
}

/// The package's two query entry points under `package_dir`: the payload
/// lays them down at `<home>/python/urna/embed/`, a checkout keeps them at
/// `rust/bridge/python/urna/embed/`.
fn lay_down(package_dir: &Path) {
    std::fs::create_dir_all(package_dir.join("embed")).unwrap();
    for name in ["searchtxt.py", "presetqry.py"] {
        let rel = Path::new("embed").join(name);
        std::fs::copy(package().join(&rel), package_dir.join(&rel)).unwrap();
    }
}

/// `cmd.output()`, retried while the binary is busy. the tests copy the
/// binary and run the copy while other tests fork in parallel: on linux a
/// child forked during the copy holds the write descriptor until its own
/// exec, and running the copy then fails with ETXTBSY (os error 26).
fn output(cmd: &mut Command) -> std::process::Output {
    for _ in 0..100 {
        match cmd.output() {
            Err(e) if e.raw_os_error() == Some(26) => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            res => return res.unwrap(),
        }
    }
    cmd.output().unwrap()
}

/// The checkout's package, relative to the repository root.
const PKG: &str = "rust/bridge/python/urna";

fn tail(parts: &[&str]) -> String {
    parts
        .iter()
        .flat_map(|p| p.split('/'))
        .collect::<PathBuf>()
        .display()
        .to_string()
}

#[test]
fn every_query_verb_finds_its_embedder_in_the_data_root() {
    let dir = scratch("data_root");
    let bin = detached_binary(&dir);
    let embed = dir.join("data").join("urna").join("python").join("urna");
    lay_down(&embed);
    let embed = embed.join("embed");
    let cases = [
        ("search-text", embed.join("searchtxt.py")),
        ("ask", embed.join("presetqry.py")),
        ("retrieve", embed.join("presetqry.py")),
    ];
    for (verb, script) in cases {
        let err = run(&bin, &dir, &dir, verb);
        assert!(
            err.contains(&script.display().to_string()),
            "{verb}: expected {}, got: {err}",
            script.display()
        );
        assert!(!err.contains("embedder script not found"), "{verb}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_a_payload_every_verb_names_the_missing_script() {
    let dir = scratch("missing");
    let bin = detached_binary(&dir);
    for (verb, script) in [
        ("search-text", tail(&[PKG, "embed", "searchtxt.py"])),
        ("ask", tail(&[PKG, "embed", "presetqry.py"])),
        ("retrieve", tail(&[PKG, "embed", "presetqry.py"])),
    ] {
        let err = run(&bin, &dir, &dir, verb);
        assert!(err.contains("embedder script not found"), "{verb}: {err}");
        assert!(err.contains(&script), "{verb}: {err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_checkout_in_the_cwd_wins_over_the_data_root() {
    let dir = scratch("checkout");
    let bin = detached_binary(&dir);
    lay_down(&dir.join("data").join("urna").join("python").join("urna"));
    let checkout = dir.join("checkout");
    lay_down(&checkout.join(tail(&[PKG])));
    // the cwd comes back canonical (/private/var on macos): match the tail.
    for (verb, rel) in [
        (
            "search-text",
            tail(&["checkout", PKG, "embed", "searchtxt.py"]),
        ),
        ("ask", tail(&["checkout", PKG, "embed", "presetqry.py"])),
    ] {
        let err = run(&bin, &dir, &checkout, verb);
        assert!(
            err.contains(&rel),
            "{verb}: expected the checkout's script, got: {err}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The files of a complete payload (mirrors `cmd::payload::REQUIRED`, which
/// this binary-crate test cannot import; the payload test pins that list to
/// what `tool/tasks/embedpack.py` ships).
const REQUIRED: [&str; 24] = [
    "VERSION",
    "python/urna/__init__.py",
    "python/urna/embed/__init__.py",
    "python/urna/embed/lexifloor.py",
    "python/urna/embed/potionqry.py",
    "python/urna/embed/potiontab.py",
    "python/urna/embed/presetqry.py",
    "python/urna/embed/searchtxt.py",
    "python/urna/embed/stbackend.py",
    "python/urna/embed/stprocess.py",
    "python/urna/embed/visionemb.py",
    "python/urna/model/__init__.py",
    "python/urna/model/catalogue.json",
    "python/urna/model/embedders.py",
    "python/urna/model/installer.py",
    "python/urna/model/modelhash.py",
    "python/urna/model/presetmap.py",
    "python/urna/model/potionb8m/README.md",
    "python/urna/model/potionb8m/config.json",
    "python/urna/model/potionb8m/model.safetensors",
    "python/urna/model/potionb8m/modules.json",
    "python/urna/model/potionb8m/special_tokens_map.json",
    "python/urna/model/potionb8m/tokenizer.json",
    "python/urna/model/potionb8m/tokenizer_config.json",
];

/// A complete payload under `home`, built from the checkout's own files
/// (the table hard-linked, not copied), minus `drop`.
fn lay_down_payload(home: &Path, drop: &str) {
    for rel in REQUIRED.iter().filter(|r| **r != drop) {
        let dst = home.join(rel);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        if *rel == "VERSION" {
            std::fs::write(&dst, format!("{}\n", env!("CARGO_PKG_VERSION"))).unwrap();
            continue;
        }
        let src = repo().join("rust").join("bridge").join(rel);
        if std::fs::hard_link(&src, &dst).is_err() {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

#[test]
fn a_payload_missing_a_module_names_the_files_and_setup_not_pip() {
    let Some(python) = ["python3", "python"].into_iter().find(|p| {
        Command::new(p)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }) else {
        eprintln!("skip: no python on PATH to run the embedder");
        return;
    };
    let dir = scratch("broken_payload");
    let bin = detached_binary(&dir);
    let home = dir.join("data").join("urna");
    lay_down_payload(&home, "python/urna/model/presetmap.py");
    let fixture = repo().join("rust/format/tests/fixtures/golden_v1_minimal.urna");
    let out = output(
        Command::new(&bin)
            .args(["ask", fixture.to_str().unwrap(), "q"])
            .current_dir(&dir)
            .env("URNA_DATA_DIR", dir.join("data"))
            .env("XDG_DATA_HOME", dir.join("xdg"))
            .env("HOME", dir.join("home"))
            .env_remove("LOCALAPPDATA")
            .env("URNA_PYTHON", python),
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        err.contains("payload") && err.contains("is incomplete"),
        "{err}"
    );
    assert!(err.contains("python/urna/model/presetmap.py"), "{err}");
    assert!(err.contains(&home.display().to_string()), "{err}");
    assert!(err.contains("urna setup"), "{err}");
    assert!(!err.contains("pip install"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}
