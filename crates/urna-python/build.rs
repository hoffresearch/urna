fn main() {
    // Use pyo3-build-config to emit Python linker flags automatically.
    pyo3_build_config::use_pyo3_cfgs();
    // On macOS, allow undefined Python symbols for extension modules.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
