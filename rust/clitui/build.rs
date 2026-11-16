//! Decides whether this binary is a release or a development build, from
//! the tag, never from a value kept by hand.
//!
//! - `URNA_RELEASE_TAG` set (the release workflow sets it to the tag it
//!   builds, `.github/buildprep.yml`): a release. The tag must be `v` plus
//!   the crate version, or the build fails.
//! - No `rust/bridge/python` next to the crate (a build from crates.io): a
//!   release.
//! - Any other build from a checkout: a development build. It gets
//!   `URNA_DEV_TREE` (the checkout's python package) and is named
//!   `<version>+dev.<sha>`, or `<version>+dev` without `.git`. Not by
//!   `git describe`: the release tags up to v0.5.4 sit on commits outside
//!   main's history, so describe names today's main after v0.2.0, and a
//!   `cargo install --git` checkout carries no tags at all.

use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    println!("cargo:rerun-if-env-changed=URNA_RELEASE_TAG");
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    if let Some(tag) = std::env::var("URNA_RELEASE_TAG")
        .ok()
        .filter(|t| !t.is_empty())
    {
        if tag != format!("v{version}") {
            eprintln!(
                "URNA_RELEASE_TAG is {tag}, but this crate is version {version}: \
                 a release build must be tagged v{version}"
            );
            return ExitCode::FAILURE;
        }
        println!("cargo:rustc-env=URNA_RELEASE_TAG={tag}");
        return ExitCode::SUCCESS;
    }

    let tree = Path::new(&manifest)
        .join("..")
        .join("bridge")
        .join("python");
    if !tree.join("urna").is_dir() {
        return ExitCode::SUCCESS;
    }
    let tree = tree.canonicalize().unwrap_or(tree);
    println!("cargo:rustc-env=URNA_DEV_TREE={}", tree.display());

    // without `.git` the name has no sha; a path that does not exist reruns
    // this script on every build, so a `git init` later is picked up.
    let dot_git = Path::new(&manifest).join("..").join("..").join(".git");
    if !dot_git.exists() {
        println!("cargo:rerun-if-changed={}", dot_git.display());
    }
    let name = match short_sha() {
        Ok(sha) => format!("{version}+dev.{sha}"),
        Err(e) => {
            println!("cargo:warning=development build named without a sha: {e}");
            format!("{version}+dev")
        }
    };
    println!("cargo:rustc-env=URNA_BUILD_NAME={name}");
    ExitCode::SUCCESS
}

/// The checkout's short sha, `git rev-parse --short HEAD` run in the crate
/// directory. The script reruns when `HEAD` moves or the branch it names
/// gets a commit (both through `--git-path`, so a linked worktree points at
/// the shared refs). No git, or no repository, is an error the caller turns
/// into the sha-less name.
fn short_sha() -> Result<String, String> {
    let sha = git(&["rev-parse", "--short", "HEAD"])?;
    if sha.is_empty() {
        return Err("git rev-parse --short HEAD printed nothing".into());
    }
    let mut watched = vec!["HEAD".to_owned()];
    if let Ok(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for name in watched {
        if let Ok(path) = git(&["rev-parse", "--git-path", &name])
            && Path::new(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    Ok(sha)
}

/// One git command's trimmed stdout, or why it failed. `GIT_OPTIONAL_LOCKS`
/// keeps a build from taking the index lock an editor may hold.
fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .map_err(|e| format!("git {}: {e}", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("git {}: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}
