//! The model catalog the payload ships (`urna/model/catalogue.json`,
//! generated from the registry with the values setup takes from upstreams:
//! the uv release, the python range, torch and the lock, the sample corpus),
//! and what the model screen reads off an entry: its download on this
//! platform, torch counted once, and its estimated minimum ram.

use serde::Deserialize;

/// One reviewed, pinned code file a model runs from its repo.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Pin {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub embedding_model: String,
    pub repo: String,
    pub revision: String,
    pub bytes: u64,
    pub model_hash: String,
    pub packages: Vec<String>,
    pub remote_code: bool,
    // the fields below came with the lock; a catalogue from before it has
    // none, and the screen leaves out what it cannot say.
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub remote_code_hashes: Vec<Pin>,
    #[serde(default)]
    pub dim: u64,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub license: String,
    /// the parameters its weights hold, read off their safetensors headers.
    #[serde(default)]
    pub params: u64,
    /// the bytes one parameter takes once loaded, by device (`cpu`, `mps`).
    #[serde(default)]
    pub dtype_bytes: std::collections::BTreeMap<String, u64>,
}

/// The device a model runs on where setup installs it: torch's cpu build
/// everywhere, which on an apple-silicon mac also drives the gpu (mps).
pub fn device(target: &str) -> &'static str {
    if target == "aarch64-macos" {
        "mps"
    } else {
        "cpu"
    }
}

impl Entry {
    /// The minimum ram this model needs on `target`, estimated: its
    /// parameters times the bytes each takes in the dtype it loads in there,
    /// plus torch's own memory on that platform. `None` when the catalogue
    /// predates the counts.
    pub fn min_ram(&self, torch: Option<&Torch>, target: &str) -> Option<u64> {
        let per = *self.dtype_bytes.get(device(target))?;
        if self.params == 0 {
            return None;
        }
        let base = torch.and_then(|t| t.rss.get(target)).copied().unwrap_or(0);
        Some(self.params * per + base)
    }

    /// What installing it downloads on `target`: its weights, plus torch
    /// when no model installed before brought it. an error when torch ships
    /// no build for `target` (every catalogue model runs on torch).
    pub fn download(
        &self,
        torch: Option<&Torch>,
        target: &str,
        has_torch: bool,
    ) -> Result<u64, String> {
        if has_torch {
            return Ok(self.bytes);
        }
        match torch {
            None => Ok(self.bytes),
            Some(t) => t
                .sizes
                .get(target)
                .map(|n| self.bytes + n)
                .ok_or_else(|| format!("torch {} ships no build for {target}", t.version)),
        }
    }

    /// The license's terms in a word, for the ones that limit use.
    pub fn terms(&self) -> Option<&'static str> {
        self.license
            .contains("-NC-")
            .then_some("non-commercial use only")
    }
}

/// `0.54 GB`.
pub fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

/// A ram estimate as setup shows it: rounded up to the next whole GB
/// (`~2 GB`), rounding to 8, 16 or 32 would hide the difference.
pub fn ram_gb(bytes: u64) -> u64 {
    bytes.div_ceil(1 << 30).max(1)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Excluded {
    pub name: String,
    pub reason: String,
}

/// One uv archive: where it is and the sha256 its release publishes.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Archive {
    pub url: String,
    pub sha256: String,
}

/// The uv release setup downloads when the machine has none, one archive
/// per platform urna ships, keyed `<arch>-<os>` (`Scan::target`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct UvRelease {
    pub version: String,
    pub archives: std::collections::BTreeMap<String, Archive>,
}

/// The python range the env is built on: the cpython minors torch has
/// wheels for (`>=3.10,<3.15`), read off torch's wheels.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct PythonRange {
    pub requires: String,
    pub torch: String,
}

/// The sample corpus setup offers: one release of a dataset on hugging
/// face, pinned to a commit, with the size and sha256 read at that commit.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct Sample {
    pub repo: String,
    pub release: String,
    pub commit: String,
    pub path: String,
    pub url: String,
    pub size: u64,
    pub sha256: String,
}

/// The torch the lock pins: its cpu index (pip's fallback installs from
/// it) and its download size on each platform it ships for, keyed like
/// `UvRelease::archives`; a platform it ships no build for has no size.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct Torch {
    pub version: String,
    pub index: String,
    pub sizes: std::collections::BTreeMap<String, u64>,
    /// torch's own memory on each platform (imported, no model), measured
    /// by the preset job.
    #[serde(default)]
    pub rss: std::collections::BTreeMap<String, u64>,
}

/// The lock beside the catalogue: exact versions of every offered model's
/// packages, for every platform, resolved together.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct LockRef {
    pub file: String,
    pub sha256: String,
}

/// A payload from 0.5.4 or older carries no `uv`, no `python` and no
/// `sample`: setup then uses the uv on PATH, lets it pick the python, as it
/// did, and offers no sample. one from before the lock carries no `torch`
/// and no `lock`: packages install unconstrained, as they did.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub uv: Option<UvRelease>,
    #[serde(default)]
    pub python: Option<PythonRange>,
    #[serde(default)]
    pub sample: Option<Sample>,
    #[serde(default)]
    pub torch: Option<Torch>,
    #[serde(default)]
    pub lock: Option<LockRef>,
    pub models: Vec<Entry>,
    pub excluded: Vec<Excluded>,
}

impl Catalog {
    pub fn parse(text: &str) -> anyhow::Result<Catalog> {
        Ok(serde_json::from_str(text)?)
    }

    /// The offered entry for a catalog name or a manifest model name; an
    /// excluded or unknown name is an error that says why.
    pub fn find(&self, name: &str) -> Result<&Entry, String> {
        if let Some(e) = self
            .models
            .iter()
            .find(|e| e.name == name || e.embedding_model == name)
        {
            return Ok(e);
        }
        if let Some(x) = self.excluded.iter().find(|x| x.name == name) {
            return Err(format!("{} is not offered: {}", x.name, x.reason));
        }
        let offered: Vec<&str> = self.models.iter().map(|e| e.name.as_str()).collect();
        Err(format!(
            "{name} is not in the model catalog (offered: {})",
            if offered.is_empty() {
                "none".to_string()
            } else {
                offered.join(", ")
            }
        ))
    }

    /// `--model` values to entries, in catalog order without repeats; `all`
    /// selects every offered model.
    pub fn select(&self, names: &[String]) -> Result<Vec<Entry>, String> {
        if names.iter().any(|n| n == "all") {
            return Ok(self.models.clone());
        }
        let mut picked = Vec::new();
        for n in names {
            let e = self.find(n)?;
            if !picked.contains(e) {
                picked.push(e.clone());
            }
        }
        picked.sort_by_key(|e| self.models.iter().position(|m| m == e));
        Ok(picked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(kind: &str, license: &str) -> Entry {
        let per = if kind == "st_multimodal" { 2 } else { 4 };
        Entry {
            name: "m".into(),
            embedding_model: "org/m".into(),
            repo: "org/m".into(),
            revision: "r".repeat(40),
            bytes: 1_190_000_000,
            model_hash: "sha256:m".into(),
            packages: vec!["torch".into()],
            remote_code: false,
            kind: kind.into(),
            remote_code_hashes: vec![],
            dim: 1024,
            modalities: vec!["text".into()],
            license: license.into(),
            params: 595_776_512,
            dtype_bytes: [("cpu".to_string(), 4), ("mps".to_string(), per)].into(),
        }
    }

    fn torch() -> Torch {
        Torch {
            version: "2.14.1".into(),
            index: "https://download.pytorch.org/whl/cpu".into(),
            sizes: [
                ("aarch64-macos".to_string(), 127_657_364),
                ("x86_64-linux".to_string(), 196_260_555),
            ]
            .into(),
            rss: [("aarch64-macos".to_string(), 448_593_920)].into(),
        }
    }

    #[test]
    fn ram_is_the_parameters_in_the_devices_dtype_plus_torch() {
        let e = entry("st_multimodal", "Apache-2.0");
        let t = torch();
        // qwen3: 596M parameters, float16 on mps, float32 on the cpu.
        assert_eq!(
            e.min_ram(Some(&t), "aarch64-macos"),
            Some(595_776_512 * 2 + 448_593_920)
        );
        assert_eq!(e.min_ram(Some(&t), "x86_64-linux"), Some(595_776_512 * 4));
        assert_eq!(ram_gb(595_776_512 * 2 + 448_593_920), 2);
        assert_eq!(ram_gb(595_776_512 * 4), 3);
        let clip = entry("open_clip", "Apache-2.0");
        assert_eq!(clip.min_ram(None, "aarch64-macos"), Some(595_776_512 * 4));
        let old = Entry {
            params: 0,
            ..entry("st_multimodal", "")
        };
        assert_eq!(old.min_ram(Some(&t), "aarch64-macos"), None);
    }

    #[test]
    fn torch_counts_once_and_a_platform_without_it_cannot_install() {
        let e = entry("st_multimodal", "Apache-2.0");
        let t = torch();
        assert_eq!(
            e.download(Some(&t), "x86_64-linux", false),
            Ok(1_190_000_000 + 196_260_555)
        );
        assert_eq!(
            e.download(Some(&t), "x86_64-linux", true),
            Ok(1_190_000_000)
        );
        let err = e.download(Some(&t), "x86_64-macos", false).unwrap_err();
        assert!(
            err.contains("torch 2.14.1 ships no build for x86_64-macos"),
            "{err}"
        );
        // already in the env, the platform does not matter.
        assert_eq!(
            e.download(Some(&t), "x86_64-macos", true),
            Ok(1_190_000_000)
        );
        assert_eq!(e.download(None, "x86_64-macos", false), Ok(1_190_000_000));
        assert_eq!(gb(1_190_000_000 + 196_260_555), "1.39 GB");
    }

    #[test]
    fn a_non_commercial_license_says_so() {
        assert_eq!(
            entry("st_multimodal", "CC-BY-NC-4.0").terms(),
            Some("non-commercial use only")
        );
        assert_eq!(entry("st_multimodal", "Apache-2.0").terms(), None);
        assert_eq!(device("aarch64-macos"), "mps");
        assert_eq!(device("x86_64-windows"), "cpu");
    }
}
