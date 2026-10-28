//! The embedder payload's contract, shared by `urna setup` (which lays it
//! down) and the query gate (which reads it to name a broken install): the
//! files a complete payload holds and where an installed one lives.

use std::path::{Path, PathBuf};

/// The payload's one folder: the `urna` package keeps its repo layout under
/// it (`python/urna/...`), beside the `VERSION` stamp.
#[cfg(feature = "tui")]
pub const DIR: &str = "python";

/// The `urna` package inside a payload home.
#[cfg(feature = "tui")]
pub const PACKAGE: &str = "python/urna";

/// The `urna` package of the payload installed under data root `root`
/// (`<root>/urna/python/urna`).
pub fn package_in(root: &Path) -> PathBuf {
    root.join("urna").join("python").join("urna")
}

/// What a complete payload holds, relative to `urna/`: every file
/// `embedpack.py` ships, since each query path imports or reads
/// one of them (the potion route reads the table's config, tokenizer and
/// weights; the registry route imports the adapters, the st and image
/// backends and modelhash, and hands a model no preset names to
/// searchtxt.py). the modules keep the package layout of
/// `rust/bridge/python/urna/` under `python/`.
/// setup refuses a payload missing one, and the scan reports an installed
/// payload missing one so setup repairs it. `tool/tests/test_embedpack.py`
/// checks that this list and the staged payload are the same set of files.
pub const REQUIRED: [&str; 26] = [
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

/// The required files missing under `home` (`<root>/urna`).
pub fn missing(home: &Path) -> Vec<&'static str> {
    REQUIRED
        .into_iter()
        .filter(|rel| !home.join(rel).is_file())
        .collect()
}

/// The python package the payload itself provides: an import error naming
/// it is a broken payload, never a package to pip install.
pub const MODULES: [&str; 1] = ["urna"];

/// The installed payload a script belongs to (`<root>/urna`), when it is
/// one: `<root>/urna/python/urna/<sub>/<name>`. a checkout's
/// `rust/bridge/python/urna/` is not a payload.
pub fn home_of(script: &Path) -> Option<PathBuf> {
    let package = script.parent()?.parent()?;
    let python = package.parent()?;
    let home = python.parent()?;
    (package.file_name()? == "urna"
        && python.file_name()? == "python"
        && home.file_name()? == "urna")
        .then(|| home.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_of_finds_an_installed_payload_and_not_a_checkout() {
        let d = std::env::temp_dir().join(format!("urna_payload_home_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let home = d.join("data").join("urna");
        std::fs::create_dir_all(home.join("python/urna/embed")).unwrap();
        std::fs::create_dir_all(d.join("rust/bridge/python/urna/embed")).unwrap();
        assert_eq!(
            home_of(&home.join("python/urna/embed/presetqry.py")),
            Some(home.clone())
        );
        assert_eq!(
            home_of(&home.join("python/urna/embed/searchtxt.py")),
            Some(home.clone())
        );
        assert_eq!(
            home_of(&d.join("rust/bridge/python/urna/embed/presetqry.py")),
            None
        );
        assert_eq!(missing(&home).len(), REQUIRED.len());
        let _ = std::fs::remove_dir_all(&d);
    }
}
