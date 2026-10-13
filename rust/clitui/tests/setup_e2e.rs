// SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
//
// SPDX-License-Identifier: MIT

//! `urna setup` end to end: the real binary against a `file://` release
//! built here (a payload tarball and its `.sha256`, the layout the release
//! workflow publishes), so the download, the checksum, the unpack and the
//! typed exit codes run exactly as they do against github. needs `curl` on
//! PATH (every ci runner has it); skips cleanly without it.
#![cfg(feature = "tui")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use flate2::Compression;
use flate2::write::GzEncoder;
use sha2::{Digest, Sha256};

const PAYLOAD: &str = "urna-embedder-payload.tar.gz";

/// `file://` url for a local dir (`file:///C:/...` on windows).
fn file_url(p: &Path) -> String {
    let s = p.display().to_string().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

fn has_curl() -> bool {
    Command::new("curl").arg("--version").output().is_ok()
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("urna_setup_e2e_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// This binary's version, the payload version `setup` asks for by default.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The files a complete payload holds besides the stamp (mirrors
/// `unpack::REQUIRED`, which this binary-crate test cannot import; a drift
/// fails the install below).
const REQUIRED: [&str; 25] = [
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
    "python/urna/model/installed.py",
    "python/urna/model/installer.py",
    "python/urna/model/modelhash.py",
    "python/urna/model/modellock.txt",
    "python/urna/model/presetmap.py",
    "python/urna/model/potionb8m/README.md",
    "python/urna/model/potionb8m/config.json",
    "python/urna/model/potionb8m/model.safetensors",
    "python/urna/model/potionb8m/modules.json",
    "python/urna/model/potionb8m/special_tokens_map.json",
    "python/urna/model/potionb8m/tokenizer.json",
    "python/urna/model/potionb8m/tokenizer_config.json",
];

/// A release dir with a complete payload stamped with this binary's
/// version; `sha` overrides the published digest (a tampered release).
fn release(dir: &Path, sha: Option<&str>) -> PathBuf {
    release_at(dir, "release", sha, VERSION, None)
}

/// A release under `dir/<name>` whose payload is stamped `version`, every
/// file's body naming that release; `drop` leaves one required file out.
fn release_at(
    dir: &Path,
    name: &str,
    sha: Option<&str>,
    version: &str,
    drop: Option<&str>,
) -> PathBuf {
    let rel = dir.join(name);
    std::fs::create_dir_all(&rel).unwrap();
    std::fs::copy(golden(), rel.join(SAMPLE)).unwrap();
    let tgz = rel.join(PAYLOAD);
    let gz = GzEncoder::new(std::fs::File::create(&tgz).unwrap(), Compression::fast());
    let mut b = tar::Builder::new(gz);
    let mut entries: Vec<(String, Vec<u8>)> = REQUIRED
        .iter()
        .filter(|f| Some(**f) != drop)
        .map(|f| (format!("urna/{f}"), body(f, version, &rel)))
        .collect();
    entries.push(("urna/VERSION".into(), format!("{version}\n").into_bytes()));
    for (name, body) in &entries {
        let (name, body) = (name.as_str(), body.as_slice());
        let mut h = tar::Header::new_gnu();
        h.set_size(body.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append_data(&mut h, name, body).unwrap();
    }
    b.into_inner().unwrap().finish().unwrap();
    let digest = hex::encode(Sha256::digest(std::fs::read(&tgz).unwrap()));
    let line = format!("{} *{PAYLOAD}\n", sha.unwrap_or(&digest));
    std::fs::write(rel.join(format!("{PAYLOAD}.sha256")), line).unwrap();
    rel
}

/// The sample corpus a test release serves in place of the hub's.
const SAMPLE: &str = "sample.urna";

/// A small valid `.urna`, the format crate's golden file.
fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../format/tests/fixtures/golden_v1_minimal.urna")
}

/// A stub file's body; the catalog is the real one, since setup reads it,
/// with its sample pointed at the copy in `rel`, so no test reaches the hub.
fn body(f: &str, version: &str, rel: &Path) -> Vec<u8> {
    if f == "python/urna/model/catalogue.json" {
        let real = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../rust/bridge/python/urna/model/catalogue.json");
        let mut cat: serde_json::Value =
            serde_json::from_slice(&std::fs::read(real).unwrap()).unwrap();
        let bytes = std::fs::read(golden()).unwrap();
        let s = &mut cat["sample"];
        s["url"] = file_url(&rel.join(SAMPLE)).into();
        s["path"] = format!("release/test/{SAMPLE}").into();
        s["size"] = bytes.len().into();
        s["sha256"] = hex::encode(Sha256::digest(&bytes)).into();
        return serde_json::to_vec_pretty(&cat).unwrap();
    }
    format!("{f}@{version}").into_bytes()
}

/// Runs the binary with the data dir and the release pointed into `dir`,
/// no interpreter pinned and no terminal (plain mode).
fn urna(dir: &Path, rel: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_urna"))
        .args(args)
        .current_dir(dir)
        .env("URNA_DATA_DIR", dir.join("data"))
        .env("URNA_RELEASE_BASE", file_url(rel))
        .env("NO_COLOR", "1")
        .env_remove("URNA_PYTHON")
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn setup_yes_installs_the_payload_from_the_release() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("ok");
    let rel = release(&d, None);
    // a venv from an earlier setup: the payload step must leave it alone.
    std::fs::create_dir_all(d.join("data/urna/venv/bin")).unwrap();
    std::fs::write(d.join("data/urna/venv/bin/marker"), b"keep").unwrap();
    let out = urna(&d, &rel, &["setup", "--yes", "--force", "--no-python"]);
    let s = text(&out);
    assert!(s.contains("ok embedder payload"), "{s}");
    assert!(d.join("data/urna/python/urna/embed/potionqry.py").is_file());
    assert!(
        d.join("data/urna/python/urna/model/modelhash.py").is_file(),
        "{s}"
    );
    assert!(
        d.join("data/urna/python/urna/embed/searchtxt.py").is_file(),
        "{s}"
    );
    assert!(d.join("data/urna/venv/bin/marker").is_file(), "{s}");
    // the stub payload has no potion table, so verify fails with a doctor
    // code (2..=6), never a setup code: the steps themselves succeeded.
    let code = out.status.code().unwrap();
    assert!((0..=6).contains(&code), "exit {code}: {s}");
    assert!(!s.contains('\x1b'), "plain mode printed an escape");
}

#[test]
fn an_upgraded_binary_replaces_an_older_payload_and_keeps_the_venv() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("upgrade");
    // an older payload without the stamp (0.5.1's had none), and a venv.
    let embed = d.join("data/urna/python/urna/embed");
    std::fs::create_dir_all(&embed).unwrap();
    std::fs::write(embed.join("potionqry.py"), b"0.5.1").unwrap();
    std::fs::create_dir_all(d.join("data/urna/venv/bin")).unwrap();
    std::fs::write(d.join("data/urna/venv/bin/marker"), b"keep").unwrap();

    // no --force: the plan sees the unstamped payload and replaces it.
    let new = release(&d, None);
    let out = urna(&d, &new, &["setup", "--yes", "--no-python"]);
    let s = text(&out);
    assert!(s.contains("ok embedder payload"), "{s}");
    let stamp = std::fs::read_to_string(d.join("data/urna/VERSION")).unwrap();
    assert_eq!(stamp.trim(), VERSION);
    assert!(d.join("data/urna/python/urna/embed/searchtxt.py").is_file());
    assert!(d.join("data/urna/venv/bin/marker").is_file());

    // the payload matches the binary now: nothing to replace.
    let again = text(&urna(&d, &new, &["setup", "--yes", "--no-python"]));
    assert!(!again.contains("ok embedder payload"), "{again}");

    // --version names another release: replaced again, venv still there.
    let pinned = text(&urna(
        &d,
        &new,
        &["setup", "--yes", "--no-python", "--version", "v9.9.9"],
    ));
    assert!(pinned.contains("ok embedder payload"), "{pinned}");
    assert!(d.join("data/urna/venv/bin/marker").is_file());
}

#[test]
fn setup_repairs_a_payload_missing_a_required_file() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("repair");
    let rel = release(&d, None);
    urna(&d, &rel, &["setup", "--yes", "--no-python"]);
    // each of these breaks a query when gone: search-text's embedder, the
    // module the potion route imports, the potion table's tokenizer.
    for gone in [
        "python/urna/embed/searchtxt.py",
        "python/urna/embed/potiontab.py",
        "python/urna/model/potionb8m/tokenizer.json",
    ] {
        std::fs::remove_file(d.join("data/urna").join(gone)).unwrap();
        // same release, no --force: the missing file alone makes setup reinstall.
        let s = text(&urna(&d, &rel, &["setup", "--yes", "--no-python"]));
        assert!(s.contains(&format!("missing {gone}")), "{gone}: {s}");
        assert!(s.contains("ok embedder payload"), "{gone}: {s}");
        assert!(
            d.join("data/urna").join(gone).is_file(),
            "{gone} not repaired"
        );
    }
}

#[test]
fn an_incomplete_release_fails_and_keeps_the_installed_payload() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("halfrel");
    let old = release_at(&d, "old", None, "0.0.1", None);
    urna(
        &d,
        &old,
        &["setup", "--yes", "--no-python", "--version", "0.0.1"],
    );
    let half = release_at(
        &d,
        "half",
        None,
        VERSION,
        Some("python/urna/embed/presetqry.py"),
    );
    let out = urna(&d, &half, &["setup", "--yes", "--no-python"]);
    assert_eq!(out.status.code(), Some(12), "{}", text(&out));
    assert!(
        text(&out).contains("python/urna/embed/presetqry.py"),
        "{}",
        text(&out)
    );
    let stamp = std::fs::read_to_string(d.join("data/urna/VERSION")).unwrap();
    assert_eq!(stamp.trim(), "0.0.1");
    let model = std::fs::read(d.join("data/urna/python/urna/embed/presetqry.py")).unwrap();
    assert_eq!(model, b"python/urna/embed/presetqry.py@0.0.1");
}

#[test]
fn a_tampered_checksum_exits_11_and_installs_nothing() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("sha");
    let rel = release(&d, Some(&"0".repeat(64)));
    let out = urna(&d, &rel, &["setup", "--yes", "--force", "--no-python"]);
    assert_eq!(out.status.code(), Some(11), "{}", text(&out));
    assert!(!d.join("data/urna/python").exists());
    assert!(text(&out).contains("does not match the release checksum"));
}

#[test]
fn a_missing_release_exits_10() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("gone");
    let out = urna(
        &d,
        &d.join("no-such-release"),
        &["setup", "--yes", "--force", "--no-python"],
    );
    assert_eq!(out.status.code(), Some(10), "{}", text(&out));
}

#[test]
fn uninstall_removes_the_payload_and_keeps_the_binary() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("rm");
    let rel = release(&d, None);
    urna(&d, &rel, &["setup", "--yes", "--force", "--no-python"]);
    assert!(d.join("data/urna/python").is_dir());
    assert!(d.join("data/urna/VERSION").is_file());
    // the uv setup downloads when a machine has none goes with the rest.
    std::fs::create_dir_all(d.join("data/urna/bin")).unwrap();
    std::fs::write(d.join("data/urna/bin/uv"), b"uv").unwrap();
    let out = urna(&d, &rel, &["setup", "--uninstall"]);
    assert_eq!(out.status.code(), Some(0));
    assert!(!d.join("data/urna/python").exists());
    assert!(!d.join("data/urna/bin").exists(), "setup's uv left behind");
    assert!(!d.join("data/urna/VERSION").exists(), "VERSION left behind");
    assert!(Path::new(env!("CARGO_BIN_EXE_urna")).is_file());
}

/// The payload an install up to 0.5.4 laid down under `<data root>/urna`
/// (layout until 0.5.4; remove with `unpack::LEGACY` in the release after next).
const LEGACY: [&str; 4] = [
    "forge",
    "embed_query.py",
    "model_fingerprint.py",
    "__pycache__",
];

fn lay_down_legacy(home: &Path) {
    std::fs::create_dir_all(home.join("forge/models")).unwrap();
    std::fs::write(home.join("forge/potion.py"), b"0.5.4").unwrap();
    std::fs::write(home.join("embed_query.py"), b"0.5.4").unwrap();
    std::fs::write(home.join("model_fingerprint.py"), b"0.5.4").unwrap();
    std::fs::create_dir_all(home.join("__pycache__")).unwrap();
    std::fs::write(
        home.join("__pycache__/embed_query.cpython-313.pyc"),
        b"0.5.4",
    )
    .unwrap();
}

#[test]
fn setup_over_the_old_layout_removes_it() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("legacy");
    let home = d.join("data/urna");
    lay_down_legacy(&home);
    std::fs::create_dir_all(home.join("venv/bin")).unwrap();
    std::fs::write(home.join("venv/bin/marker"), b"keep").unwrap();
    let rel = release(&d, None);
    let s = text(&urna(&d, &rel, &["setup", "--yes", "--no-python"]));
    assert!(s.contains("ok embedder payload"), "{s}");
    for name in LEGACY {
        assert!(!home.join(name).exists(), "{name} left behind: {s}");
    }
    assert!(home.join("python/urna/embed/potionqry.py").is_file(), "{s}");
    assert!(home.join("venv/bin/marker").is_file(), "{s}");
}

#[test]
fn uninstall_removes_the_old_layout_too() {
    let d = scratch("legacy_rm");
    let home = d.join("data/urna");
    lay_down_legacy(&home);
    let out = urna(&d, &d, &["setup", "--uninstall"]);
    assert_eq!(out.status.code(), Some(0), "{}", text(&out));
    for name in LEGACY {
        assert!(
            !home.join(name).exists(),
            "{name} left behind: {}",
            text(&out)
        );
    }
}

#[cfg(unix)]
#[test]
fn uninstall_names_an_old_layout_it_could_not_remove() {
    use std::os::unix::fs::PermissionsExt;
    let d = scratch("legacy_locked");
    let home = d.join("data/urna");
    lay_down_legacy(&home);
    // a home the user cannot write: nothing in it can be removed.
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o555)).unwrap();
    let out = urna(&d, &d, &["setup", "--uninstall"]);
    std::fs::set_permissions(&home, std::fs::Permissions::from_mode(0o755)).unwrap();
    let all = format!("{}{}", text(&out), String::from_utf8_lossy(&out.stderr));
    assert_ne!(out.status.code(), Some(0), "{all}");
    assert!(all.contains("could not remove"), "{all}");
    assert!(
        !all.contains("removed "),
        "reported a removal that failed: {all}"
    );
    for name in LEGACY {
        assert!(home.join(name).exists(), "{name}: {all}");
    }
}

#[test]
fn a_bare_urna_without_a_terminal_prints_help_and_exits_2() {
    let d = scratch("bare");
    let out = urna(&d, &d, &[]);
    assert_eq!(out.status.code(), Some(2));
    let all = format!("{}{}", text(&out), String::from_utf8_lossy(&out.stderr));
    assert!(all.contains("Usage: urna"), "{all}");
    assert!(all.contains("setup"));
}

#[test]
fn doctor_through_a_pipe_has_no_escapes_and_names_setup() {
    let d = scratch("doc");
    let out = Command::new(env!("CARGO_BIN_EXE_urna"))
        .arg("doctor")
        .current_dir(&d)
        .env("URNA_DATA_DIR", d.join("data"))
        .env("URNA_PYTHON", "/urna/no/such/python")
        .output()
        .unwrap();
    let s = text(&out);
    assert_eq!(out.status.code(), Some(2), "{s}");
    assert!(!s.contains('\x1b'));
    assert!(s.contains("urna setup"), "{s}");
}

/// `--model` names catalog models; setup refuses, with the reason and before
/// anything is downloaded, a model the catalog leaves out, repo code that was
/// not separately allowed, and packages for an env other than its own.
#[test]
fn setup_refuses_models_the_catalog_or_the_consent_does_not_cover() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("models");
    let rel = release(&d, None);
    // every other data root (and the checkout) out of reach: setup must
    // install into its own data dir's venv and nowhere else.
    let urna = |d: &Path, rel: &Path, args: &[&str]| {
        output(isolated(d, rel, args).env_remove("URNA_PYTHON"))
    };
    // no payload yet: the catalog arrives with it, the name is checked then.
    let out = urna(
        &d,
        &rel,
        &["setup", "--yes", "--no-python", "--model", "potion"],
    );
    let s = text(&out);
    assert_eq!(out.status.code(), Some(14), "{s}");
    assert!(s.contains("ok embedder payload"), "{s}");
    assert!(
        s.contains("potion is not offered: ships inside the payload"),
        "{s}"
    );
    // installed: the plan itself names it and skips the step.
    let out = urna(
        &d,
        &rel,
        &["setup", "--yes", "--no-python", "--model", "potion"],
    );
    let s = text(&out);
    assert_eq!(out.status.code(), Some(14), "{s}");
    assert!(s.contains("skip models: potion is not offered"), "{s}");

    // a catalog with a model that runs repo code.
    let coded = r#"{"schema": 1, "excluded": [], "models": [{"name": "coded",
        "embedding_model": "org/coded", "repo": "org/coded", "revision": "cccccccc",
        "bytes": 1000000, "model_hash": "sha256:c", "packages": [], "remote_code": true}]}"#;
    std::fs::write(d.join("data/urna/python/urna/model/catalogue.json"), coded).unwrap();
    let out = urna(
        &d,
        &rel,
        &["setup", "--yes", "--no-python", "--model", "coded"],
    );
    let s = text(&out);
    assert_eq!(out.status.code(), Some(14), "{s}");
    assert!(s.contains("--allow-remote-code coded"), "{s}");
    // allowed, but there is no managed venv to put anything in.
    let args = [
        "setup",
        "--yes",
        "--no-python",
        "--model",
        "coded",
        "--allow-remote-code",
        "coded",
    ];
    let out = urna(&d, &rel, &args);
    let s = text(&out);
    assert_eq!(out.status.code(), Some(14), "{s}");
    assert!(s.contains("managed venv is not built yet"), "{s}");
    // a pinned interpreter elsewhere: packages never go there.
    let out = output(isolated(&d, &rel, &args).env("URNA_PYTHON", "/elsewhere/python3"));
    let s = text(&out);
    assert!(s.contains("URNA_PYTHON pins /elsewhere/python3"), "{s}");
    assert!(!d.join("data/urna/venv").exists(), "{s}");
}

/// The binary run from a copy outside the checkout, with HOME and every
/// data root inside `dir`.
fn isolated(dir: &Path, rel: &Path, args: &[&str]) -> Command {
    let bin = dir.join("bin");
    let src = PathBuf::from(env!("CARGO_BIN_EXE_urna"));
    let exe = bin.join(src.file_name().unwrap());
    if !exe.is_file() {
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::copy(&src, &exe).unwrap();
    }
    let mut c = Command::new(exe);
    c.args(args)
        .current_dir(dir)
        .env("URNA_DATA_DIR", dir.join("data"))
        .env("XDG_DATA_HOME", dir.join("xdg"))
        .env("HOME", dir.join("home"))
        .env_remove("LOCALAPPDATA")
        .env("URNA_RELEASE_BASE", file_url(rel))
        .env("NO_COLOR", "1");
    c
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

/// A fake uv as the catalogue's archive (`uv-<t>/uv` in a `.tar.gz`): a
/// shell script that logs its arguments to `<data>/uv.log`, makes `venv`
/// lay down a python that answers like 3.12 with numpy and tokenizers, and
/// passes `pip install`. enough to prove setup downloads the uv the
/// catalogue names, checks it, and builds the env on the range.
#[cfg(unix)]
fn fake_uv_release(dir: &Path, sha: Option<&str>) -> String {
    let script = format!(
        r#"#!/bin/sh
echo "$@" >> "{log}"
if [ "$1" = venv ]; then
  for a; do last="$a"; done
  /bin/mkdir -p "$last/bin"
  printf '#!/bin/sh\ncase "$1" in --version) echo "Python 3.12.0";; *) echo "numpy 2.0 · tokenizers 0.20";; esac\n' > "$last/bin/python"
  /bin/chmod 755 "$last/bin/python"
fi
exit 0
"#,
        log = dir.join("uv.log").display()
    );
    let tgz = dir.join("uv-fake.tar.gz");
    let gz = GzEncoder::new(std::fs::File::create(&tgz).unwrap(), Compression::fast());
    let mut b = tar::Builder::new(gz);
    let mut h = tar::Header::new_gnu();
    h.set_size(script.len() as u64);
    h.set_mode(0o755);
    h.set_cksum();
    b.append_data(&mut h, "uv-fake/uv", script.as_bytes())
        .unwrap();
    b.into_inner().unwrap().finish().unwrap();
    let digest = hex::encode(Sha256::digest(std::fs::read(&tgz).unwrap()));
    let target = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);
    format!(
        r#"{{"schema": 1, "excluded": [], "models": [],
        "uv": {{"version": "0.13.0", "archives": {{"{target}": {{"url": "{}", "sha256": "{}"}}}}}},
        "python": {{"requires": ">=3.10,<3.15", "torch": "2.14.1"}}}}"#,
        file_url(&tgz),
        sha.unwrap_or(&digest)
    )
}

/// `isolated`, with PATH holding curl and nothing else: no uv, no python.
#[cfg(unix)]
fn bare_machine(dir: &Path, rel: &Path, args: &[&str]) -> Command {
    let tools = dir.join("tools");
    std::fs::create_dir_all(&tools).unwrap();
    let curl = String::from_utf8(
        Command::new("sh")
            .args(["-c", "command -v curl"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let link = tools.join("curl");
    if !link.exists() {
        std::os::unix::fs::symlink(curl.trim(), &link).unwrap();
    }
    let mut c = isolated(dir, rel, args);
    c.env("PATH", &tools).env_remove("URNA_PYTHON");
    c
}

#[cfg(unix)]
#[test]
fn a_machine_without_uv_or_python_gets_the_catalogues_uv_and_range() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("uvfetch");
    let rel = release(&d, None);
    let data = d.join("data");
    // the payload, then a catalogue that names the fake uv.
    let out = output(&mut isolated(&d, &rel, &["setup", "--yes", "--no-python"]));
    assert!(text(&out).contains("ok embedder payload"), "{}", text(&out));
    let catalogue = data.join("urna/python/urna/model/catalogue.json");
    std::fs::write(&catalogue, fake_uv_release(&data, None)).unwrap();
    // an env from before, on a python outside the range: rebuilt, not reused.
    let old = data.join("urna/venv/bin/python");
    std::fs::create_dir_all(old.parent().unwrap()).unwrap();
    std::fs::write(&old, "#!/bin/sh\necho Python 3.15.0\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::write(data.join("urna/venv/marker"), b"old").unwrap();

    let out = output(&mut bare_machine(
        &d,
        &rel,
        &["setup", "--yes", "--no-payload"],
    ));
    let s = text(&out);
    assert!(
        s.contains("runs python 3.15, outside python 3.10 to 3.14"),
        "the plan says the env is rebuilt: {s}"
    );
    assert!(s.contains("uv 0.13.0 · sha256 ok"), "{s}");
    assert!(s.contains("ok python env"), "{s}");
    assert!(data.join("urna/bin/uv").is_file(), "{s}");
    assert!(
        !data.join("urna/venv/marker").exists(),
        "the old env was reused: {s}"
    );
    let log = std::fs::read_to_string(data.join("uv.log")).unwrap();
    assert!(
        log.contains("venv --allow-existing --quiet --python >=3.10,<3.15"),
        "{log}"
    );

    // a uv whose bytes are not the catalogue's: refused, nothing installed,
    // and with no python on PATH the step names the fix.
    std::fs::remove_dir_all(data.join("urna/bin")).unwrap();
    std::fs::remove_dir_all(data.join("urna/venv")).unwrap();
    std::fs::write(&catalogue, fake_uv_release(&data, Some(&"0".repeat(64)))).unwrap();
    let out = output(&mut bare_machine(
        &d,
        &rel,
        &["setup", "--yes", "--no-payload"],
    ));
    let s = text(&out);
    assert_eq!(out.status.code(), Some(14), "{s}");
    assert!(s.contains("does not match the catalogue's sha256"), "{s}");
    assert!(s.contains("no python 3.10 to 3.14 is on PATH"), "{s}");
    assert!(s.contains("install uv (https://docs.astral.sh/uv/)"), "{s}");
    assert!(!data.join("urna/bin/uv").exists(), "{s}");
}

#[test]
fn the_sample_is_saved_checked_and_other_bytes_are_refused() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("sample");
    let rel = release(&d, None);
    let s = text(&urna(&d, &rel, &["setup", "--yes", "--no-python"]));
    assert!(s.contains("ok sample corpus"), "{s}");
    let saved = d.join("data/urna/sample").join(SAMPLE);
    assert_eq!(
        std::fs::read(&saved).unwrap(),
        std::fs::read(golden()).unwrap()
    );

    // the release serves other bytes than the catalogue's sha256.
    let d = scratch("sample_bad");
    let rel = release(&d, None);
    std::fs::write(rel.join(SAMPLE), b"not the sample").unwrap();
    let s = text(&urna(&d, &rel, &["setup", "--yes", "--no-python"]));
    assert!(s.contains("fail(16) sample corpus"), "{s}");
    assert!(s.contains("does not match the catalogue's sha256"), "{s}");
    let left: Vec<_> = std::fs::read_dir(d.join("data/urna/sample"))
        .map(|r| r.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "left behind: {left:?}");
}

#[test]
fn no_sample_skips_the_step() {
    if !has_curl() {
        eprintln!("skip: no curl on PATH");
        return;
    }
    let d = scratch("no_sample");
    let rel = release(&d, None);
    let s = text(&urna(
        &d,
        &rel,
        &["setup", "--yes", "--no-python", "--no-sample"],
    ));
    assert!(s.contains("skip sample corpus"), "{s}");
    assert!(!s.contains("==> sample corpus"), "{s}");
    assert!(!d.join("data/urna/sample").exists(), "{s}");
}
