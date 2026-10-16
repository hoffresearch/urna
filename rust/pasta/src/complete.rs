//! Type-ahead over card names, English and Portuguese. Tiers first (exact,
//! prefix, word prefix, fuzzy); inside a name tier the more played card
//! (EDHREC rank) leads, inside the fuzzy tier the better match does.
//! One suggestion per name: the playable card wins over its art-series twin
//! and the many tokens that share a name collapse into one line.

use std::collections::HashSet;
use std::sync::Mutex;

use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32String};

use crate::card::{Card, Kind};

#[derive(Clone, Debug)]
pub struct Suggestion {
    pub ordinal: usize,
    pub name: String,
    pub pt_name: Option<String>,
    /// The Portuguese name matched better than the English one.
    pub via_pt: bool,
    pub score: u32,
}

struct Entry {
    ordinal: usize,
    folded: String,
    hay: Utf32String,
    via_pt: bool,
    playable: bool,
    /// `u32::MAX - edhrec_rank`, so higher sorts first; 0 when unranked.
    popularity: u32,
}

/// (playable, tier, popularity in name tiers, score, -len): sorted descending.
type Rank = (bool, u8, u32, u32, i64);

pub struct Completer {
    entries: Vec<Entry>,
    matcher: Mutex<Matcher>,
}

/// Lowercase with the accents dropped: `Relâmpago` -> `relampago`.
pub fn fold(s: &str) -> String {
    s.chars()
        .map(nucleo_matcher::chars::normalize)
        .flat_map(char::to_lowercase)
        .collect()
}

impl Completer {
    pub fn new(cards: &[Card]) -> Self {
        let mut entries = Vec::with_capacity(cards.len() * 2);
        // art-series repeat a real card's name and alchemy cards are digital:
        // neither belongs in a name list.
        for c in cards
            .iter()
            .filter(|c| !matches!(c.kind, Kind::ArtSeries | Kind::Alchemy))
        {
            let names =
                std::iter::once((&c.name, false)).chain(c.pt_name.as_ref().map(|p| (p, true)));
            for (name, via_pt) in names {
                entries.push(Entry {
                    ordinal: c.ordinal,
                    folded: fold(name),
                    hay: Utf32String::from(name.as_str()),
                    via_pt,
                    playable: c.playable(),
                    popularity: c.edhrec_rank.map_or(0, |r| u32::MAX - r),
                });
            }
        }
        Self {
            entries,
            matcher: Mutex::new(Matcher::new(Config::DEFAULT)),
        }
    }

    pub fn complete(&self, cards: &[Card], query: &str, k: usize) -> Vec<Suggestion> {
        let q = fold(query.trim());
        if q.is_empty() || k == 0 {
            return Vec::new();
        }
        let pattern = Pattern::parse(&q, CaseMatching::Ignore, Normalization::Smart);
        let mut matcher = match self.matcher.lock() {
            Ok(m) => m,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut scored: Vec<(Rank, &Entry)> = Vec::new();
        for e in &self.entries {
            let Some(score) = pattern.score(e.hay.slice(..), &mut matcher) else {
                continue;
            };
            let tier = if e.folded == q {
                3
            } else if e.folded.starts_with(&q) {
                2
            } else if e.folded.split([' ', ',', '-']).any(|w| w.starts_with(&q)) {
                1
            } else {
                0
            };
            let popularity = if tier > 0 { e.popularity } else { 0 };
            scored.push((
                (
                    e.playable,
                    tier,
                    popularity,
                    score,
                    -(e.folded.len() as i64),
                ),
                e,
            ));
        }
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.ordinal.cmp(&b.1.ordinal)));

        let mut seen: HashSet<&str> = HashSet::new();
        let mut out = Vec::with_capacity(k);
        for ((_, _, _, score, _), e) in scored {
            let card = &cards[e.ordinal];
            if !seen.insert(card.name.as_str()) {
                continue;
            }
            out.push(Suggestion {
                ordinal: e.ordinal,
                name: card.name.clone(),
                pt_name: card.pt_name.clone(),
                via_pt: e.via_pt,
                score,
            });
            if out.len() == k {
                break;
            }
        }
        out
    }
}
