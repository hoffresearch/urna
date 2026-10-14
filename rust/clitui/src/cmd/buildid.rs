//! What this binary is, as `build.rs` decided at compile time: a release,
//! which fetches the payload of its own tag, or a development build, which
//! installs the payload of the checkout it was built from.

#[cfg(feature = "tui")]
use std::path::Path;

/// The checkout's python package (`rust/bridge/python`) a development build
/// was compiled from; `None` for a release. Only setup reads it.
#[cfg(feature = "tui")]
pub const DEV_TREE: Option<&str> = option_env!("URNA_DEV_TREE");

/// The name `--version` and setup show: the crate version for a release,
/// `<version>+dev.<sha>` for a development build (`+dev` without `.git`).
pub const NAME: &str = match option_env!("URNA_BUILD_NAME") {
    Some(name) => name,
    None => env!("CARGO_PKG_VERSION"),
};

#[cfg(feature = "tui")]
pub fn dev_tree() -> Option<&'static Path> {
    DEV_TREE.map(Path::new)
}
