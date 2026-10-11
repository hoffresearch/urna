//! Why a query could not be embedded or gated, as one typed value with the
//! fix that works for it. `embed_gate` raises these inside the
//! `anyhow::Error` it returns: the cli prints the long form, and the
//! explorer's ask tab downcasts and prints `short`. the embedder scripts name
//! two of the causes on a stable stderr line (`urna-needs: <pip spec>...`,
//! `urna-fetch: <model>`); the rest come from the gate itself.

use std::fmt;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedFailure {
    /// the embedder script is not where the resolution ladder looked.
    ScriptMissing { script: PathBuf },
    /// the installed payload lacks files the embedder needs (`home` when the
    /// script lives in one; the file list, or the payload module python
    /// could not import).
    PayloadIncomplete {
        home: Option<PathBuf>,
        missing: Vec<String>,
    },
    /// python packages the model needs do not import in `python`.
    DepsMissing {
        python: String,
        packages: Vec<String>,
    },
    /// the model is not in the local cache, and queries run offline.
    WeightsMissing { model: String },
    /// the embedder cannot serve this corpus: no preset or backend for its
    /// model, a refused preset, or a name or dim the manifest disagrees with.
    ModelIncompatible { model: String, detail: String },
    /// the embedder's model fingerprint is not the one the corpus was built
    /// with: the hits would be cosine-valid and wrong.
    HashMismatch {
        corpus: String,
        embedder: String,
        fingerprint: String,
    },
    /// anything else the embedder reported.
    Failed { status: String, detail: String },
}

impl EmbedFailure {
    /// Reads a failed embedder run: its exit status and stderr, the model it
    /// was asked for and the interpreter it ran under.
    pub fn from_embedder(status: &str, stderr: &str, model: &str, python: &str) -> Self {
        let lines: Vec<&str> = stderr.lines().map(str::trim).collect();
        let tagged = |tag: &str| {
            lines
                .iter()
                .find_map(|l| l.strip_prefix(tag).map(str::trim))
        };
        if let Some(pkgs) = tagged("urna-needs:").filter(|p| !p.is_empty()) {
            return Self::DepsMissing {
                python: python.to_string(),
                packages: pkgs.split_whitespace().map(String::from).collect(),
            };
        }
        if let Some(m) = tagged("urna-fetch:").filter(|m| !m.is_empty()) {
            return Self::WeightsMissing {
                model: m.to_string(),
            };
        }
        // the potion embedder imports numpy and tokenizers at the top; a
        // missing one surfaces as python's own ModuleNotFoundError. a module
        // the payload itself provides is a broken payload, never a package.
        if let Some(name) = lines.iter().find_map(|l| missing_module(l))
            && super::payload::MODULES.contains(&name.as_str())
        {
            return Self::PayloadIncomplete {
                home: None,
                missing: vec![format!("python module {name}")],
            };
        }
        if let Some(name) = lines.iter().find_map(|l| missing_module(l)) {
            return Self::DepsMissing {
                python: python.to_string(),
                packages: vec![name],
            };
        }
        let error = lines
            .iter()
            .rev()
            .find_map(|l| l.strip_prefix("error:").map(str::trim))
            .map(String::from);
        if lines
            .iter()
            .any(|l| l.starts_with("error: model asset missing"))
        {
            return Self::WeightsMissing {
                model: model.to_string(),
            };
        }
        match error {
            Some(detail) => Self::ModelIncompatible {
                model: model.to_string(),
                detail,
            },
            None => Self::Failed {
                status: status.to_string(),
                detail: lines
                    .iter()
                    .rev()
                    .find(|l| !l.is_empty())
                    .unwrap_or(&"")
                    .to_string(),
            },
        }
    }

    /// One line for a status bar: the cause and where the fix is.
    #[cfg_attr(not(feature = "tui"), allow(dead_code))]
    pub fn short(&self) -> String {
        match self {
            Self::ScriptMissing { .. } => {
                "the embedder script is not installed here (esc, then s runs setup)".into()
            }
            Self::PayloadIncomplete { missing, .. } => format!(
                "the embedder payload is incomplete, missing {} (esc, then s repairs it)",
                missing.join(", ")
            ),
            Self::DepsMissing { packages, .. } => {
                format!(
                    "the query model needs python packages: {}",
                    packages.join(" ")
                )
            }
            Self::WeightsMissing { model } => format!(
                "{model} is not in the local cache; run the query once with URNA_ALLOW_DOWNLOAD=1"
            ),
            Self::ModelIncompatible { detail, .. } => format!("this corpus's model: {detail}"),
            Self::HashMismatch { .. } => {
                "model_hash mismatch: the local model is not the one the corpus was built with"
                    .into()
            }
            Self::Failed { detail, .. } => format!("the embedder failed: {detail}"),
        }
    }
}

/// `ModuleNotFoundError: No module named 'numpy'` -> `numpy`.
fn missing_module(line: &str) -> Option<String> {
    let rest = line.split("No module named ").nth(1)?;
    let name = rest.trim().trim_matches(|c| c == '\'' || c == '"');
    let top = name.split('.').next()?;
    (!top.is_empty()).then(|| top.to_string())
}

impl fmt::Display for EmbedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScriptMissing { script } => write!(
                f,
                "embedder script not found: {}\nhint: `urna setup` installs the offline \
                 embedder payload; --embedder points at a script by hand",
                script.display()
            ),
            Self::PayloadIncomplete { home, missing } => {
                let at = home
                    .as_ref()
                    .map(|h| format!(" at {}", h.display()))
                    .unwrap_or_default();
                write!(
                    f,
                    "the embedder payload{at} is incomplete, missing {}\n\
                     hint: `urna setup` repairs it (it reinstalls the payload; the venv stays)",
                    missing.join(", ")
                )
            }
            Self::DepsMissing { python, packages } => {
                let pkgs = packages.join(" ");
                write!(
                    f,
                    "the query model needs python packages that {python} cannot import: {pkgs}\n\
                     hint: install them in that env: `uv pip install --python {python} {pkgs}` \
                     (or `{python} -m pip install {pkgs}`); URNA_PYTHON picks another interpreter"
                )
            }
            Self::WeightsMissing { model } => write!(
                f,
                "model {model} is not in the local cache, and queries run offline\n\
                 hint: run the same command once with URNA_ALLOW_DOWNLOAD=1 to fetch it; \
                 later queries stay offline"
            ),
            Self::ModelIncompatible { model, detail } => write!(
                f,
                "the embedder cannot serve this corpus's model {model}: {detail}"
            ),
            Self::HashMismatch {
                corpus,
                embedder,
                fingerprint,
            } => write!(
                f,
                "model_hash mismatch: corpus was built with {corpus}, embedder reports {embedder}\n\
                 fingerprint reported by embedder: {fingerprint}\n\
                 hint: --model-path PATH to point at the exact snapshot, or rebuild the corpus \
                 with the model you intend to use. --skip-model-hash-check covers the legacy \
                 placeholder only, not a mismatch."
            ),
            Self::Failed { status, detail } => {
                write!(f, "embedder failed (status={status}): {detail}")
            }
        }
    }
}

impl std::error::Error for EmbedFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(stderr: &str) -> EmbedFailure {
        EmbedFailure::from_embedder("exit status: 4", stderr, "org/m", "/v/bin/python")
    }

    #[test]
    fn the_stable_lines_name_packages_and_weights() {
        assert_eq!(
            read("error: x\nurna-needs: torch sentence-transformers>=5.7\n"),
            EmbedFailure::DepsMissing {
                python: "/v/bin/python".into(),
                packages: vec!["torch".into(), "sentence-transformers>=5.7".into()],
            }
        );
        assert_eq!(
            read("error: not cached\nurna-fetch: org/m\n"),
            EmbedFailure::WeightsMissing {
                model: "org/m".into()
            }
        );
    }

    #[test]
    fn a_python_import_error_is_a_missing_package() {
        let tb = "Traceback (most recent call last):\n  File \"x.py\"\n\
                  ModuleNotFoundError: No module named 'numpy.core'\n";
        assert_eq!(
            read(tb),
            EmbedFailure::DepsMissing {
                python: "/v/bin/python".into(),
                packages: vec!["numpy".into()],
            }
        );
    }

    #[test]
    fn a_payload_module_that_does_not_import_is_never_a_pip_package() {
        for module in ["urna.embed.potiontab", "urna.model.modelhash", "urna"] {
            let tb = format!("ModuleNotFoundError: No module named '{module}'\n");
            let f = read(&tb);
            assert!(
                matches!(f, EmbedFailure::PayloadIncomplete { .. }),
                "{module}: {f:?}"
            );
            assert!(!f.to_string().contains("pip install"), "{f}");
            assert!(
                f.to_string().contains("urna setup") && f.short().contains("repairs"),
                "{f}"
            );
        }
    }

    #[test]
    fn a_refusal_is_incompatible_and_noise_is_a_plain_failure() {
        let refused = read(
            "error: preset 'wemm-2b' runs remote model code; opt in with URNA_ALLOW_REMOTE_CODE",
        );
        assert!(
            matches!(&refused, EmbedFailure::ModelIncompatible { detail, .. } if detail.contains("URNA_ALLOW_REMOTE_CODE")),
            "{refused:?}"
        );
        assert_eq!(
            read("error: model asset missing: /x/ViT-B-32.bin"),
            EmbedFailure::WeightsMissing {
                model: "org/m".into()
            }
        );
        let noise = read("Segmentation fault\n");
        assert!(matches!(noise, EmbedFailure::Failed { .. }), "{noise:?}");
    }

    #[test]
    fn every_message_names_its_fix() {
        let s = EmbedFailure::ScriptMissing {
            script: "/d/urna/searchtxt.py".into(),
        };
        assert!(s.to_string().starts_with("embedder script not found"));
        assert!(s.to_string().contains("urna setup") && s.short().contains("setup"));
        let d = read("urna-needs: sentence-transformers");
        assert!(
            d.to_string()
                .contains("uv pip install --python /v/bin/python sentence-transformers")
        );
        let w = read("urna-fetch: org/m");
        assert!(
            w.to_string().contains("URNA_ALLOW_DOWNLOAD=1")
                && w.short().contains("URNA_ALLOW_DOWNLOAD=1")
        );
        let h = EmbedFailure::HashMismatch {
            corpus: "sha256:a".into(),
            embedder: "sha256:b".into(),
            fingerprint: "{}".into(),
        };
        assert!(
            h.to_string().starts_with("model_hash mismatch") && h.short().contains("model_hash")
        );
    }
}
