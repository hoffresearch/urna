//! The ask tab: `urna ask` with a screen. The query embeds offline through
//! the same routed embedder and model gate (`embed_gate::embed_and_search`)
//! on a worker thread, against the corpus the loader opened once; hits come
//! back with their stored canonical text (tier-1, the bytes `cite` returns)
//! and, when their source is an image blob, the image. Typing offers the
//! corpus's labels; tab takes one and asks for it, that label first. A
//! corpus with chunk_scalars takes the search syntax its schema declares
//! (`t:creature c:r pow>=3 f:pauper`): the predicates narrow the
//! suggestions and the hits before anything is scored, and a line of
//! predicates alone browses the kept chunks by rank.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use ratatui_cheese::input::InputState;
use urna_engine::scalars::{Bitset, parse};

use super::labels;
use super::media::Media;
use super::offer::{Need, Offer};
use super::session::Session;
use crate::cmd::embed_failure::EmbedFailure;
use crate::cmd::embed_gate;

#[derive(Clone, Debug)]
pub struct Answer {
    pub citation: String,
    pub source: String,
    pub score: f32,
    pub text: String,
    /// the image blob the source points at, when it is one.
    pub image: Option<usize>,
    /// found by its label, not by the search: `score` is not a cosine.
    pub by_name: bool,
    /// the chunk's index in the file.
    pub chunk: usize,
    /// `hybrid` from the search, `name` from the label, `browse` from the
    /// kept chunks by rank.
    pub via: &'static str,
}

/// A failed query: the words for the toast, and the model to install when
/// the failure is one the install panel can fix.
#[derive(Clone, Debug)]
pub struct Failure {
    pub text: String,
    pub need: Option<Need>,
}

type Reply = Result<(Vec<Answer>, f64), Failure>;

/// How many hits to show.
const K: usize = 20;
/// How many kept chunks to browse with no text.
const BROWSE: usize = 60;

pub struct Ask {
    pub input: InputState,
    pub answers: Vec<Answer>,
    pub sel: usize,
    pub scroll: u16,
    pub asked: Option<String>,
    pub took_ms: f64,
    /// the last type-ahead's time.
    pub typed_ms: f64,
    /// the last query's error, in words a person can act on.
    pub failed: Option<String>,
    pub(super) pending: Option<(Receiver<Reply>, Instant)>,
    /// the install panel, open over the tab.
    pub offer: Option<Offer>,
    pub media: Media,
    /// the corpus the last query ran on, for the image pane.
    pub session: Option<Arc<Session>>,
    /// labels offered for what is typed, and the one ↑↓ points at.
    pub suggestions: Vec<String>,
    pub sug_sel: usize,
    /// the input changed since the last ask: the list shows suggestions.
    pub typing: bool,
    /// a label taken with tab: hits whose first line is it go first.
    pinned: Option<String>,
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
            typed_ms: 0.0,
            failed: None,
            pending: None,
            offer: None,
            media: Media::default(),
            session: None,
            suggestions: Vec::new(),
            sug_sel: 0,
            typing: false,
            pinned: None,
        }
    }
}

/// What a line said: the free text and the chunks the predicates keep
/// (`None` when there were no predicates).
#[derive(Debug)]
pub struct Line {
    pub text: String,
    pub keep: Option<Bitset>,
    /// the predicate terms, for the status line.
    pub terms: Vec<String>,
}

/// Parse `input` with the corpus's schema and resolve its predicates.
pub fn line(session: &Session, input: &str) -> Result<Line, String> {
    let parsed = parse(input, session.schema()).map_err(|e| e.to_string())?;
    let keep = if parsed.filter.is_empty() {
        None
    } else {
        Some(session.rt.keep(&parsed.filter).map_err(|e| e.to_string())?)
    };
    Ok(Line {
        text: parsed.text,
        keep,
        terms: parsed.terms,
    })
}

impl Ask {
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Starts a query against the opened corpus on a worker thread. With
    /// chunk_scalars, predicates alone browse the kept chunks by rank.
    pub fn submit(&mut self, session: Arc<Session>) {
        let input = self.input.value().trim().to_string();
        if input.is_empty() || self.busy() {
            return;
        }
        let (tx, rx) = channel();
        let s = Arc::clone(&session);
        let query = input.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let _ = tx.send(run_line(&s, &query).map(|a| (a, t0.elapsed().as_secs_f64() * 1e3)));
        });
        self.session = Some(session);
        self.asked = Some(input);
        self.typing = false;
        self.pending = Some((rx, Instant::now()));
    }

    /// Runs the last query again (after an install fixed what it lacked).
    pub fn retry(&mut self, session: Arc<Session>) {
        if let Some(q) = self.asked.clone() {
            self.input.set_value(q);
        }
        self.submit(session);
    }

    /// The input changed: offer the labels that match its text, under its
    /// predicates.
    pub fn retype(&mut self, session: Option<&Session>) {
        self.typing = true;
        self.pinned = None;
        self.sug_sel = 0;
        let t0 = Instant::now();
        self.suggestions = session
            .and_then(|s| {
                let l = line(s, self.input.value()).ok()?;
                Some(match &l.keep {
                    Some(keep) => s.labels.suggest_where(&l.text, 30, &|i| keep.get(i)),
                    None => s.labels.suggest_where(&l.text, 30, &|_| true),
                })
            })
            .unwrap_or_default();
        self.typed_ms = t0.elapsed().as_secs_f64() * 1e3;
    }

    /// Tab: the pointed label becomes the text, asked with it first; the
    /// predicates typed stay.
    pub fn take_suggestion(&mut self, session: Arc<Session>) {
        let Some(label) = self.suggestions.get(self.sug_sel).cloned() else {
            return;
        };
        let terms: Vec<String> = line(&session, self.input.value())
            .map(|l| l.terms)
            .unwrap_or_default();
        let value = if terms.is_empty() {
            label.clone()
        } else {
            format!("{label} {}", terms.join(" "))
        };
        self.input.set_value(value);
        self.input.end();
        self.pinned = Some(label);
        self.submit(session);
    }

    /// The suggestions are what the list shows.
    pub fn suggesting(&self) -> bool {
        self.typing && !self.suggestions.is_empty()
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
        Some(reply.map(|(mut answers, ms)| {
            self.failed = None;
            // a hit whose label answers the query by name goes before one
            // that only scores close: the label taken with tab first, then a
            // label the text spells out, then the rest; stable, so rank order
            // holds inside each. among equals a ranked chunk leads an
            // unranked one (a card leads its art-series twin).
            let pinned = self.pinned.take();
            let session = self.session.clone();
            let asked = self.asked.clone().unwrap_or_default();
            let text = session
                .as_deref()
                .and_then(|s| line(s, &asked).ok())
                .map(|l| l.text)
                .unwrap_or(asked);
            let q = labels::fold(&text);
            answers.sort_by_key(|a| {
                let first = a.text.lines().next().map(str::trim).unwrap_or("");
                let by_tab = pinned.as_deref() == Some(first);
                let t = labels::tier(&labels::fold(first), &q);
                let ranked = session.as_deref().and_then(|s| s.rank(a.chunk)).is_some();
                std::cmp::Reverse((by_tab, if t >= 2 { t } else { 0 }, ranked))
            });
            self.answers = answers;
            self.took_ms = ms;
            self.sel = 0;
            self.scroll = 0;
            self.answers.len()
        }))
    }
}

/// Parse and run one line: free text searches under the predicates,
/// predicates alone browse.
fn run_line(session: &Session, input: &str) -> Result<Vec<Answer>, Failure> {
    let l = line(session, input).map_err(|text| Failure { text, need: None })?;
    if l.text.trim().is_empty() {
        return Ok(browse(session, l.keep.as_ref()));
    }
    run_query(session, &l.text, l.keep.as_ref())
}

/// Search `query` over the chunks `keep` accepts (all when `None`).
pub(super) fn run_query(
    session: &Session,
    query: &str,
    keep: Option<&Bitset>,
) -> Result<Vec<Answer>, Failure> {
    embed_gate::embed_and_search_where(&session.rt, query, K as i32, None, None, None, keep)
        .map(|result| {
            let index = index_of(session);
            let mut answers: Vec<Answer> = result
                .hits
                .iter()
                .filter_map(|h| {
                    let chunk = index(&h.chunk_id)?;
                    Some(Answer {
                        citation: h.citation_id.clone(),
                        source: h.source_uri.clone(),
                        score: h.score,
                        text: session.texts.get(&h.chunk_id).cloned().unwrap_or_default(),
                        image: session.image_of(&h.source_uri),
                        by_name: false,
                        chunk,
                        via: "hybrid",
                    })
                })
                .collect();
            // a chunk whose label is the query goes first, found by the
            // search or not, when the predicates keep it.
            let named: Vec<usize> = session
                .named(query)
                .iter()
                .copied()
                .filter(|&i| keep.is_none_or(|k| k.get(i)))
                .collect();
            answers.retain(|a| !named.contains(&a.chunk));
            let extra: Vec<Answer> = named
                .iter()
                .filter_map(|&i| answer_at(session, i, 1.0, true, "name"))
                .collect();
            answers.splice(0..0, extra);
            answers
        })
        .map_err(|e| {
            let manifest = session
                .rt
                .inspect_json()
                .ok()
                .and_then(|j| serde_json::from_str::<serde_json::Value>(&j).ok())
                .map(|info| info["manifest"].clone());
            Failure {
                text: explain(&e),
                need: need(&e, manifest.as_ref()),
            }
        })
}

/// chunk_id -> chunk index: a map when the corpus is large enough to pay
/// for it, a scan of the ids otherwise.
fn index_of(session: &Session) -> impl Fn(&str) -> Option<usize> + '_ {
    let ids = session.rt.chunk_ids();
    let map: Option<std::collections::HashMap<&str, usize>> = (ids.len() > 256).then(|| {
        ids.iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect()
    });
    move |id: &str| match &map {
        Some(m) => m.get(id).copied(),
        None => ids.iter().position(|x| x == id),
    }
}

/// The hit for chunk `i` as the file stores it.
fn answer_at(
    session: &Session,
    i: usize,
    score: f32,
    by_name: bool,
    via: &'static str,
) -> Option<Answer> {
    let id = session.rt.chunk_ids().get(i)?;
    let span = session.rt.span(i)?;
    Some(Answer {
        citation: format!("urna://{}/{id}", session.rt.content_hash()),
        image: session.image_of(&span.source_uri),
        source: span.source_uri.clone(),
        score,
        text: session.texts.get(id).cloned().unwrap_or_default(),
        by_name,
        chunk: i,
        via,
    })
}

/// No text: the kept chunks by rank (the rank-role column, unranked last).
fn browse(session: &Session, keep: Option<&Bitset>) -> Vec<Answer> {
    let n = session.rt.chunk_ids().len();
    let mut kept: Vec<usize> = (0..n).filter(|&i| keep.is_none_or(|k| k.get(i))).collect();
    kept.sort_by_key(|&i| (session.rank(i).unwrap_or(i64::MAX), i));
    kept.into_iter()
        .take(BROWSE)
        .filter_map(|i| answer_at(session, i, 0.0, false, "browse"))
        .collect()
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
    use std::path::PathBuf;

    #[test]
    fn the_hit_list_scrolls_to_keep_the_selection_visible() {
        let answers = (0..10)
            .map(|i| Answer {
                citation: format!("urna://c/{i}"),
                source: "s".into(),
                score: 1.0 - i as f32 / 10.0,
                text: format!("hit number {i}"),
                image: None,
                by_name: false,
                chunk: i,
                via: "hybrid",
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
        let session = Arc::new(Session::open(&golden).expect("golden opens"));
        ask.submit(session);
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

    #[test]
    fn a_prefix_the_schema_lacks_is_a_message_not_a_panic() {
        let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../format/tests/fixtures/golden_v1_minimal.urna");
        let session = Session::open(&golden).expect("golden opens");
        let err = line(&session, "t:creature bolt").expect_err("no schema");
        assert!(err.contains("prefix"), "{err}");
        let ok = line(&session, "lightning bolt").expect("free text");
        assert_eq!(
            (ok.text.as_str(), ok.keep.is_none()),
            ("lightning bolt", true)
        );
    }
}
