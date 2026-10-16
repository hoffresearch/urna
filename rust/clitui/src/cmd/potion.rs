//! The potion query embedder in process (#205): `urna_neural::Potion` found
//! through the same ladder as the script, loaded once per process and
//! shared by every query after.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};

pub use urna_neural::Potion;

/// The table dir next to the potion script: `<pkg>/embed/potionqry.py` ->
/// `<pkg>/model/potionb8m`.
pub fn table_dir(script: &Path) -> Option<PathBuf> {
    let dir = script.parent()?.parent()?.join("model").join("potionb8m");
    dir.join("model.safetensors").is_file().then_some(dir)
}

/// The table dir and its loaded embedder, the one this process keeps.
type Loaded = Option<(PathBuf, Arc<Potion>)>;

/// The loaded embedder for `dir`, shared by every query of the process.
pub fn shared(dir: &Path) -> Result<Arc<Potion>> {
    static CACHE: OnceLock<Mutex<Loaded>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut slot = cell.lock().map_err(|_| anyhow!("potion cache poisoned"))?;
    if let Some((d, p)) = slot.as_ref()
        && d == dir
    {
        return Ok(Arc::clone(p));
    }
    let p = Arc::new(Potion::open(dir)?);
    *slot = Some((dir.to_path_buf(), Arc::clone(&p)));
    Ok(p)
}

#[cfg(test)]
mod tests {
    //! parity against the reference embedder script, run when a python with
    //! numpy and tokenizers is around (the repo venv, then `python3`), and
    //! skipped with a note otherwise.

    use super::*;
    use std::process::Command;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn table() -> Option<PathBuf> {
        let dir = root().join("rust/bridge/python/urna/model/potionb8m");
        let bytes = std::fs::read(dir.join("model.safetensors")).ok()?;
        (!bytes.starts_with(b"version https://git-lfs")).then_some(dir)
    }

    fn python() -> Option<String> {
        let venv = root().join(".venv/bin/python");
        let candidates = [venv.to_string_lossy().into_owned(), "python3".into()];
        candidates.into_iter().find(|py| {
            Command::new(py)
                .args(["-c", "import numpy, tokenizers"])
                .output()
                .is_ok_and(|o| o.status.success())
        })
    }

    fn reference(py: &str, text: &str) -> (String, Vec<f32>) {
        let out = Command::new(py)
            .arg(root().join("rust/bridge/python/urna/embed/potionqry.py"))
            .arg("potion-base-8M")
            .arg(text)
            .output()
            .expect("spawn the reference embedder");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
        let hash = v["model_hash"].as_str().expect("model_hash").to_string();
        let vector = v["vector"]
            .as_array()
            .expect("vector")
            .iter()
            .map(|x| x.as_f64().expect("f64") as f32)
            .collect();
        (hash, vector)
    }

    #[test]
    fn vectors_and_model_hash_match_the_reference_script() {
        let (Some(dir), Some(py)) = (table(), python()) else {
            eprintln!("skip potion parity: no table or no python with numpy+tokenizers");
            return;
        };
        let potion = Potion::open(&dir).expect("potion opens");
        let long_word = "x".repeat(150);
        let texts = [
            "Lightning Bolt deals 3 damage to any target.",
            "Relâmpago: cause 3 pontos de dano a qualquer alvo.",
            &long_word,
            "😀 日本語 한국어 Ωμέγα",
            "zzzzzq qqqzz",
            "hello,world!!!don't(stop)",
        ];
        let mut worst = 0f32;
        for text in texts {
            let (hash, theirs) = reference(&py, text);
            assert_eq!(potion.model_hash(), hash, "model_hash on {text:?}");
            let ours = potion.embed(text).expect("embed");
            assert_eq!(ours.len(), theirs.len(), "dim on {text:?}");
            for (a, b) in ours.iter().zip(&theirs) {
                worst = worst.max((a - b).abs());
            }
            assert!(
                ours.iter().zip(&theirs).all(|(a, b)| (a - b).abs() <= 1e-6),
                "{text:?} differs from the reference beyond 1e-6"
            );
        }
        eprintln!("potion parity: worst component difference {worst:e}");
        // the script refuses an empty query before embedding, so the empty
        // text's contract (the unk row, as `embed_texts` maps it) is checked
        // against the unk token itself.
        assert_eq!(
            potion.embed("").expect("empty"),
            potion.embed("[UNK]").expect("unk")
        );
    }
}
