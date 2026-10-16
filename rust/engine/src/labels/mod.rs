//! The labels a corpus offers to a type-ahead (card names, titles,
//! diagnoses), folded and sorted, stored with front coding: inside a block
//! of 16 each entry keeps only what differs from the one before it. Two
//! lists: the labels, and every word a label holds after its first, as the
//! rest of the label from that word on. A prefix of a label or of a word is
//! then a binary search over the block heads and a walk over one range,
//! never a scan; the scan, a subsequence match over every label, is left
//! for when the exact matches do not fill the list. Nothing here knows what
//! a label stands for; the caller ranks the hits with what it knows.

mod coded;
mod fold;
mod fuzzy;

use coded::Coded;
pub use fold::fold;
pub use fuzzy::{alias, score, tier};

/// A label the query matched: which one, what it names, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    /// the label's position in the sorted order it was built from.
    pub id: usize,
    pub ordinal: usize,
    /// 3 the whole label, 2 a prefix, 1 a word prefix, 0 a subsequence.
    pub tier: u8,
    pub score: u32,
    /// the folded label's length in bytes, for "shorter first".
    pub len: usize,
}

/// Where a word starts in a label: the label's id and the byte offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Word {
    id: u32,
    at: u16,
}

/// The labels, in order, with their prefixes shared.
#[derive(Default)]
pub struct Index {
    labels: Coded<u32>,
    words: Coded<Word>,
    /// each label's folded length in bytes.
    lens: Vec<u16>,
    /// where each label's trailing `(…)` alias starts, `u16::MAX` for none.
    alias_at: Vec<u16>,
    /// the lowercase ascii letters each label holds, one bit per letter.
    masks: Vec<u32>,
}

crate::shape_debug!(Index { show labels, show words, len lens, len alias_at, len masks });

/// `from_sorted` refused its input: the label at `at` sorts before the one
/// before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsorted {
    pub at: usize,
}

impl std::fmt::Display for Unsorted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "labels are not sorted at {}", self.at)
    }
}

impl std::error::Error for Unsorted {}

impl Index {
    /// Builds the index over labels already folded and sorted, each with the
    /// ordinal it names; a label's id is its position here.
    pub fn from_sorted<'a>(
        labels: impl IntoIterator<Item = (&'a str, usize)>,
    ) -> Result<Self, Unsorted> {
        let mut lens = Vec::new();
        let mut alias_at = Vec::new();
        let mut masks = Vec::new();
        let mut words: Vec<(String, Word)> = Vec::new();
        let mut sorted = Vec::new();
        for (id, (label, ordinal)) in labels.into_iter().enumerate() {
            lens.push(clamp16(label.len()));
            alias_at.push(alias(label).map_or(u16::MAX, |a| clamp16(label.len() - a.len() - 1)));
            masks.push(letters(label));
            let id32 = u32::try_from(id).unwrap_or(u32::MAX);
            for at in word_starts(label) {
                words.push((
                    label[at..].to_string(),
                    Word {
                        id: id32,
                        at: clamp16(at),
                    },
                ));
            }
            sorted.push((label, u32::try_from(ordinal).unwrap_or(u32::MAX)));
        }
        let labels = Coded::from_sorted(sorted.iter().map(|(l, o)| (*l, *o)))?;
        words.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
        let words = Coded::from_sorted(words.iter().map(|(w, at)| (w.as_str(), *at)))?;
        Ok(Self {
            labels,
            words,
            lens,
            alias_at,
            masks,
        })
    }

    pub fn len(&self) -> usize {
        self.lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    pub fn ordinal(&self, id: usize) -> Option<usize> {
        self.labels.payload(id).map(|o| o as usize)
    }

    /// The label with `id`, decoded from its block.
    pub fn label(&self, id: usize) -> Option<String> {
        self.labels.get(id)
    }

    /// The ids of the labels that start with `q` (already folded).
    pub fn prefix(&self, q: &str) -> std::ops::Range<usize> {
        self.labels.prefix(q)
    }

    /// The labels `keep` accepts that `q` (not yet folded) is the whole of,
    /// a prefix of, or a prefix of one of their words: tiers 3, 2 and 1,
    /// found by binary search, with a label at most once.
    pub fn exact(&self, q: &str, keep: &dyn Fn(usize) -> bool) -> Vec<Hit> {
        let q = fold(q.trim());
        if q.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let whole = self.labels.prefix(&q);
        for id in whole.clone() {
            if let Some(hit) = self.hit(id, 0, &q, keep) {
                out.push(hit);
            }
        }
        // a label with two words that start with q is one hit, at its first
        // such word: the word index sorts by word, so the two can sit any
        // distance apart in it.
        let mut first_at: std::collections::HashMap<usize, u16> = Default::default();
        let mut order = Vec::new();
        for i in self.words.prefix(&q) {
            let Some(w) = self.words.payload(i) else {
                continue;
            };
            let id = w.id as usize;
            if whole.contains(&id) {
                continue;
            }
            first_at
                .entry(id)
                .and_modify(|at| *at = (*at).min(w.at))
                .or_insert_with(|| {
                    order.push(id);
                    w.at
                });
        }
        for id in order {
            if let Some(hit) = self.hit(id, first_at[&id] as usize, &q, keep) {
                out.push(hit);
            }
        }
        out
    }

    /// Every label `keep` accepts that `q` (not yet folded) matches as a
    /// subsequence, in id order, each with its tier; the scan the caller
    /// runs when `exact` did not fill the list.
    pub fn fuzzy(&self, q: &str, keep: &dyn Fn(usize) -> bool) -> Vec<Hit> {
        let q = fold(q.trim());
        if q.is_empty() {
            return Vec::new();
        }
        let want = letters(&q);
        let mut out = Vec::new();
        self.labels.walk(0, |id, label, ordinal| {
            let ordinal = ordinal as usize;
            if self.masks[id] & want != want || !keep(ordinal) {
                return true;
            }
            if let Some(score) = score(label, &q) {
                out.push(Hit {
                    id,
                    ordinal,
                    tier: tier(label, &q),
                    score,
                    len: label.len(),
                });
            }
            true
        });
        out
    }

    /// `exact` and then `fuzzy`, each label once, in no particular order.
    pub fn matches(&self, q: &str, keep: &dyn Fn(usize) -> bool) -> Vec<Hit> {
        let mut out = self.exact(q, keep);
        let seen: std::collections::HashSet<usize> = out.iter().map(|h| h.id).collect();
        out.extend(
            self.fuzzy(q, keep)
                .into_iter()
                .filter(|h| !seen.contains(&h.id)),
        );
        out
    }

    /// The hit for `q` found contiguous at byte `at` of label `id`.
    fn hit(&self, id: usize, at: usize, q: &str, keep: &dyn Fn(usize) -> bool) -> Option<Hit> {
        let ordinal = self.labels.payload(id)? as usize;
        if !keep(ordinal) {
            return None;
        }
        let len = self.lens[id] as usize;
        let alias_at = self.alias_at[id] as usize;
        let end = at + q.len();
        let tier = if at == 0 && (end == len || end + 2 == alias_at) {
            3
        } else if at == 0 {
            2
        } else if at == alias_at {
            if end + 1 == len { 3 } else { 2 }
        } else {
            1
        };
        Some(Hit {
            id,
            ordinal,
            tier,
            score: fuzzy::score_contiguous(q, at),
            len,
        })
    }
}

/// The byte offsets where a word starts inside `label`, after the first.
fn word_starts(label: &str) -> impl Iterator<Item = usize> + '_ {
    label
        .char_indices()
        .filter(|(_, c)| fuzzy::separator(*c))
        .map(|(i, c)| i + c.len_utf8())
        .filter(|&i| i < label.len() && !label[i..].starts_with(fuzzy::separator))
}

/// One bit per lowercase ascii letter present.
fn letters(s: &str) -> u32 {
    s.bytes()
        .filter(u8::is_ascii_lowercase)
        .fold(0, |m, b| m | 1 << (b - b'a'))
}

fn clamp16(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(labels: &[&str]) -> Index {
        let mut folded: Vec<(String, usize)> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (fold(l), i))
            .collect();
        folded.sort();
        Index::from_sorted(folded.iter().map(|(l, o)| (l.as_str(), *o))).unwrap_or_default()
    }

    fn tiers(idx: &Index, hits: Vec<Hit>) -> Vec<(usize, u8)> {
        let mut v: Vec<(usize, u8)> = hits.iter().map(|h| (h.ordinal, h.tier)).collect();
        v.sort_unstable();
        assert!(v.len() <= idx.len());
        v
    }

    #[test]
    fn labels_come_back_whole_across_blocks_and_the_prefix_range_is_exact() {
        let names: Vec<String> = (0..100)
            .map(|i| format!("lightning {} bolt (raio {i})", i % 7))
            .chain(["aether vial", "bolt of light", "zzz"].map(String::from))
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let idx = index(&refs);
        assert_eq!(idx.len(), 103);
        let mut sorted: Vec<String> = names.iter().map(|n| fold(n)).collect();
        sorted.sort();
        for (id, want) in sorted.iter().enumerate() {
            assert_eq!(idx.label(id).as_deref(), Some(want.as_str()));
        }
        let r = idx.prefix("lightning 3");
        assert_eq!(
            r.len(),
            sorted
                .iter()
                .filter(|l| l.starts_with("lightning 3"))
                .count()
        );
        assert!(
            r.clone()
                .all(|id| idx.label(id).is_some_and(|l| l.starts_with("lightning 3")))
        );
        assert_eq!(idx.prefix("zzz"), 102..103);
        assert_eq!(idx.prefix("a"), 0..1);
        assert_eq!(idx.prefix("nothing"), 0..0);
        assert_eq!(idx.prefix(""), 0..0);
        assert!(Index::from_sorted([("b", 0), ("a", 1)]).is_err());
        assert!(Index::from_sorted(std::iter::empty()).is_ok_and(|i| i.is_empty()));
        // `raio 7` and `raio 70` to `raio 79`: the alias prefix of each.
        assert_eq!(idx.exact("raio 7", &|_| true).len(), 11);
        assert_eq!(idx.exact("bolt", &|_| true).len(), 101);
    }

    #[test]
    fn exact_finds_whole_prefix_alias_and_word_without_a_scan() {
        let idx = index(&[
            "Lightning Bolt (Raio)",
            "Lightning",
            "Relâmpago Vivo",
            "Bolt of Light",
            "Fire // Ice",
        ]);
        let all = |_: usize| true;
        let t = |q: &str| tiers(&idx, idx.exact(q, &all));
        assert_eq!(t("RAIO"), vec![(0, 3)]);
        assert_eq!(t("ra"), vec![(0, 2)]);
        assert_eq!(t("lightning"), vec![(0, 2), (1, 3)]);
        assert_eq!(t("lightning bolt"), vec![(0, 3)]);
        assert_eq!(t("bolt"), vec![(0, 1), (3, 2)]);
        assert_eq!(t("light"), vec![(0, 2), (1, 2), (3, 1)]);
        assert_eq!(t("relampago vivo"), vec![(2, 3)]);
        assert_eq!(t("vivo"), vec![(2, 1)]);
        assert_eq!(t("ice"), vec![(4, 1)]);
        assert!(t("bolt light").is_empty());
        assert!(t("   ").is_empty());
        assert_eq!(idx.exact("lightning", &|o| o != 1).len(), 1);
        let hits = idx.exact("light", &all);
        let whole = hits.iter().find(|h| h.ordinal == 1).map(|h| h.score);
        let word = hits.iter().find(|h| h.ordinal == 3).map(|h| h.score);
        assert!(whole > word, "{whole:?} {word:?}");
    }

    #[test]
    fn a_label_two_words_match_is_one_hit_however_far_apart_they_index() {
        // `shock` and `sword of shock` sort with a hundred `sl…` words
        // between them in the word index.
        let names: Vec<String> = (0..100)
            .map(|i| format!("filler sl{i:03}"))
            .chain(["x sword of shock".to_string()])
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let idx = index(&refs);
        let hits = idx.exact("s", &|_| true);
        assert_eq!(hits.len(), 101);
        let sword: Vec<&Hit> = hits.iter().filter(|h| h.ordinal == 100).collect();
        assert_eq!(sword.len(), 1);
        let first = Index::from_sorted([("x sword", 0)])
            .expect("one label")
            .exact("s", &|_| true);
        assert_eq!(sword[0].score, first[0].score, "the first word wins");
    }

    #[test]
    fn fuzzy_scans_for_subsequences_and_matches_merges_the_two() {
        let idx = index(&[
            "Lightning Bolt (Raio)",
            "Lightning",
            "Relâmpago Vivo",
            "Bolt of Light",
        ]);
        let all = |_: usize| true;
        assert_eq!(tiers(&idx, idx.fuzzy("bolt light", &all)), vec![(3, 0)]);
        // `relampago vivo` holds r, a, i, o in order too: a subsequence.
        let hits = idx.fuzzy("raio", &all);
        assert_eq!(tiers(&idx, hits.clone()), vec![(0, 3), (2, 0)]);
        let bolt = hits.iter().find(|h| h.ordinal == 0).copied();
        assert!(bolt.is_some_and(|b| hits.iter().all(|h| h.score <= b.score)));
        let merged = idx.matches("raio", &all);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.iter().filter(|h| h.ordinal == 0).count(), 1);
        assert!(idx.fuzzy("qqq", &all).is_empty());
        assert!(idx.matches("", &all).is_empty());
    }
}
