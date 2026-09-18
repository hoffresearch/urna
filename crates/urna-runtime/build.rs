//! detect whether the compiling rustc is new enough for the aarch64 NEON
//! f16 intrinsics (`float16x4_t` / `vcvt_f32_f16`, stable since 1.94). when
//! it is, emit `cfg(neon_f16)` so the f16 dot product uses the vectorized
//! path; on older toolchains the dispatcher falls back to the scalar f16
//! kernel and the workspace still builds at its declared msrv (1.85).

use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(neon_f16)");
    // the cfg only gates aarch64 code; skip the rustc probe elsewhere.
    let aarch64 = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "aarch64");
    if aarch64 && rustc_minor().is_some_and(|m| m >= 94) {
        println!("cargo::rustc-cfg=neon_f16");
    }
}

/// minor version of the rustc cargo invokes us with, e.g. 96 for "1.96.0".
fn rustc_minor() -> Option<u32> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(rustc).arg("--version").output().ok()?;
    let version = String::from_utf8(out.stdout).ok()?;
    // "rustc 1.96.0 (ac68faa20 2026-05-25)" and "1.96.0-nightly" both yield 96.
    version
        .split_whitespace()
        .nth(1)?
        .split('.')
        .nth(1)?
        .parse()
        .ok()
}
