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
use std::process::ExitCode;

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
    let git = Path::new(&manifest).join("..").join("..").join(".git");
    if !git.exists() {
        println!("cargo:rerun-if-changed={}", git.display());
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

/// The checkout's short sha, from vergen. Its instructions (the reruns on
/// `.git/HEAD` and the current ref) pass through to cargo; fail_on_error
/// makes a failed `git rev-parse` an error instead of vergen's placeholder.
fn short_sha() -> anyhow::Result<String> {
    let git = vergen_gitcl::GitclBuilder::default().sha(true).build()?;
    let mut out = Vec::new();
    vergen_gitcl::Emitter::default()
        .fail_on_error()
        .add_instructions(&git)?
        .emit_to(&mut out)?;
    let out = String::from_utf8(out)?;
    print!("{out}");
    out.lines()
        .find_map(|l| l.strip_prefix("cargo:rustc-env=VERGEN_GIT_SHA="))
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("vergen emitted no VERGEN_GIT_SHA"))
}
