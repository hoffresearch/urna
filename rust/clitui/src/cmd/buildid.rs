//! What this binary is, as `build.rs` decided at compile time: a release,
//! which fetches the payload of its own tag, or a development build, which
//! installs the payload of the checkout it was built from.

/// The name `--version` and setup show: the crate version for a release,
/// `<version>+dev.<sha>` for a development build (`+dev` without `.git`).
pub const NAME: &str = match option_env!("URNA_BUILD_NAME") {
    Some(name) => name,
    None => env!("CARGO_PKG_VERSION"),
};
