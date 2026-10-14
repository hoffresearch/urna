//! The ask tab: `urna ask` with a screen. The query embeds offline through
//! the same routed embedder and model gate (`embed_gate::embed_and_search`)
//! on a worker thread; hits come back with their stored canonical text
//! (tier-1, the bytes `cite` returns). left: hits with the exact-rerank
//! score as a bar; right: the selected hit's text, scrollable.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use ratatui_cheese::input::InputState;

use super::offer::{Need, Offer};
use crate::cmd::embed_failure::EmbedFailure;
use crate::cmd::{agent, embed_gate};

pub struct Answer {
    pub citation: String,
    pub source: String,
    pub score: f32,
    pub text: String,
}

/// A failed query: the words for the toast, and the model to install when
/// the failure is one the install panel can fix.
#[derive(Clone, Debug)]
pub struct Failure {
    pub text: String,
    pub need: Option<Need>,
}

type Reply = Result<(Vec<Answer>, f64), Failure>;

pub struct Ask {
    pub input: InputState,
    pub answers: Vec<Answer>,
    pub sel: usize,
    pub scroll: u16,
    pub asked: Option<String>,
    pub took_ms: f64,
    /// the last query's error, in words a person can act on.
    pub failed: Option<String>,
    pub(super) pending: Option<(Receiver<Reply>, Instant)>,
    /// the install panel, open over the tab.
    pub offer: Option<Offer>,
}

impl Default for Ask {
    fn default() -> Self {
        let mut input = InputState::new();
        input.set_focused(true);
        Self {
            input,
            answers: Vec::new(),
            sel: 0,
            scroll: 0,
            asked: None,
            took_ms: 0.0,
            failed: None,
            pending: None,
            offer: None,
        }
    }
}

impl Ask {
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Starts a query against `file` on a worker thread.
    pub fn submit(&mut self, file: PathBuf) {
        let q = self.input.value().trim().to_string();
        if q.is_empty() || self.busy() {
            return;
        }
        let (tx, rx) = channel();
        let query = q.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let _ =
                tx.send(run_query(&file, &query).map(|a| (a, t0.elapsed().as_secs_f64() * 1e3)));
        });
        self.asked = Some(q);
        self.pending = Some((rx, Instant::now()));
    }

    /// Runs the last query again (after an install fixed what it lacked).
    pub fn retry(&mut self, file: PathBuf) {
        if let Some(q) = self.asked.clone() {
            self.input.set_value(q);
        }
        self.submit(file);
    }

    /// Folds a finished query in; returns the failure when it failed.
    pub fn poll(&mut self) -> Option<Result<usize, Failure>> {
        let (rx, _) = self.pending.as_ref()?;
        let reply = rx.try_recv().ok()?;
        self.pending = None;
        if let Err(e) = &reply {
            self.failed = Some(e.text.clone());
            self.answers.clear();
        }
        Some(reply.map(|(answers, ms)| {
            self.failed = None;
            self.answers = answers;
            self.took_ms = ms;
            self.sel = 0;
            self.scroll = 0;
            self.answers.len()
        }))
    }
}

fn run_query(file: &PathBuf, query: &str) -> Result<Vec<Answer>, Failure> {
    let mut manifest = None;
    (|| -> anyhow::Result<Vec<Answer>> {
        let rt = urna_engine::MmapUrnaFile::open(file)?;
        let info: serde_json::Value = serde_json::from_str(&rt.inspect_json()?)?;
        manifest = Some(info["manifest"].clone());
        let result = embed_gate::embed_and_search(&rt, query, 10, None, None, None)?;
        let texts = agent::retrieve::canonical_texts(file)?;
        let by_id: std::collections::HashMap<&str, &str> = texts
            .iter()
            .map(|(i, t)| (i.as_str(), t.as_str()))
            .collect();
        Ok(result
            .hits
            .iter()
            .map(|h| Answer {
                citation: h.citation_id.clone(),
                source: h.source_uri.clone(),
                score: h.score,
                text: by_id
                    .get(h.chunk_id.as_str())
                    .copied()
                    .unwrap_or("")
                    .to_string(),
            })
            .collect())
    })()
    .map_err(|e| Failure {
        text: explain(&e),
        need: need(&e, manifest.as_ref()),
    })
}

/// The corpus model to offer for install: only for the two failures an
/// install fixes (packages missing, weights missing), named by the manifest.
fn need(err: &anyhow::Error, manifest: Option<&serde_json::Value>) -> Option<Need> {
    let fixable = matches!(
        err.downcast_ref::<EmbedFailure>(),
        Some(EmbedFailure::DepsMissing { .. } | EmbedFailure::WeightsMissing { .. })
    );
    let m = manifest.filter(|_| fixable)?;
    Some(Need {
        model: m["embedding_model"].as_str()?.to_string(),
        corpus_hash: m["model_hash"].as_str()?.to_string(),
    })
}

/// The engine's error, turned toward what fixes it from this screen: a
/// typed embed failure says its own cause; an interpreter that cannot even
/// start points at setup; anything else is its first line.
pub fn explain(err: &anyhow::Error) -> String {
    if let Some(f) = err.downcast_ref::<EmbedFailure>() {
        return f.short();
    }
    let all = format!("{err:#}");
    if all.contains("failed to spawn embedder") {
        return "no python interpreter can run the embedder (esc, then s runs setup)".into();
    }
    all.lines().next().unwrap_or(&all).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn the_hit_list_scrolls_to_keep_the_selection_visible() {
        let answers = (0..10)
            .map(|i| Answer {
                citation: format!("urna://c/{i}"),
                source: "s".into(),
                score: 1.0 - i as f32 / 10.0,
                text: format!("hit number {i}"),
            })
            .collect();
        let mut ask = Ask {
            answers,
            sel: 9,
            ..Ask::default()
        };
        let area = Rect::new(0, 0, 100, 16);
        let mut buf = Buffer::empty(area);
        super::super::hits::render(&mut buf, area, &mut ask, "·");
        let screen: String = (0..16)
            .map(|y| {
                (0..100)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect();
        assert!(screen.contains("hit number 9"));
        assert!(!screen.contains("hit number 0"));
    }

    #[test]
    fn explain_names_each_cause_and_its_fix() {
        let typed = |f: EmbedFailure| explain(&anyhow::Error::new(f));
        let script = typed(EmbedFailure::ScriptMissing {
            script: "/d/urna/python/urna/embed/presetqry.py".into(),
        });
        assert!(
            script.contains("not installed") && script.contains("setup"),
            "{script}"
        );
        let deps = typed(EmbedFailure::from_embedder(
            "exit status: 4",
            "urna-needs: sentence-transformers",
            "m",
            "/v/bin/python",
        ));
        assert!(deps.contains("sentence-transformers"), "{deps}");
        let weights = typed(EmbedFailure::WeightsMissing {
            model: "org/m".into(),
        });
        assert!(
            weights.contains("org/m") && weights.contains("URNA_ALLOW_DOWNLOAD"),
            "{weights}"
        );
        let hash = typed(EmbedFailure::HashMismatch {
            corpus: "a".into(),
            embedder: "b".into(),
            fingerprint: "{}".into(),
        });
        assert!(hash.starts_with("model_hash mismatch"), "{hash}");
        let spawn = explain(&anyhow::anyhow!(
            "failed to spawn embedder: no such file (x)"
        ));
        assert!(spawn.contains("setup"), "{spawn}");
        assert_eq!(explain(&anyhow::anyhow!("first\nsecond")), "first");
    }

    #[test]
    fn only_missing_packages_or_weights_open_the_install_panel() {
        let manifest = serde_json::json!({"embedding_model": "org/m", "model_hash": "sha256:m"});
        let deps = anyhow::Error::new(EmbedFailure::DepsMissing {
            python: "/v/bin/python".into(),
            packages: vec!["sentence-transformers".into()],
        });
        assert_eq!(
            need(&deps, Some(&manifest)),
            Some(Need {
                model: "org/m".into(),
                corpus_hash: "sha256:m".into()
            })
        );
        let weights = anyhow::Error::new(EmbedFailure::WeightsMissing {
            model: "org/m".into(),
        });
        assert!(need(&weights, Some(&manifest)).is_some());
        let hash = anyhow::Error::new(EmbedFailure::HashMismatch {
            corpus: "a".into(),
            embedder: "b".into(),
            fingerprint: "{}".into(),
        });
        assert_eq!(need(&hash, Some(&manifest)), None);
        assert_eq!(need(&deps, None), None);
    }

    #[test]
    fn a_query_without_an_embedder_fails_with_words_not_a_panic() {
        let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../format/tests/fixtures/golden_v1_minimal.urna");
        let mut ask = Ask::default();
        ask.input.set_value("anything".into());
        ask.submit(golden);
        let mut got = None;
        for _ in 0..500 {
            if let Some(r) = ask.poll() {
                got = Some(r);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        // the golden fixture carries a placeholder model, so the gate or the
        // embedder refuses it; either way the ask tab gets a message.
        assert!(got.is_some());
        let area = Rect::new(0, 0, 100, 30);
        let mut buf = Buffer::empty(area);
        super::super::hits::render(&mut buf, area, &mut ask, "·");
    }
}
