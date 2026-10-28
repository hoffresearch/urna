//! Type-ahead for the ask tab over the corpus's own labels: the first line
//! of each chunk's stored text (a card name, a title, a diagnosis), or the
//! schema's label columns (a name and its translation) when the corpus
//! carries fields.
//! The engine's label index folds, stores and matches them; this module
//! ranks the hits by tier first (the whole label, a prefix of it, a prefix
//! of one of its words, then a fuzzy match), then by the score, then
//! shorter. With fields, a ranked label leads its unranked twin (a card's
//! art series), the better rank leads inside a name tier, and the many
//! tokens that share a name collapse into one line.

use std::collections::HashSet;

use urna_engine::labels::Index;
pub use urna_engine::labels::{alias, fold, tier};

use urna_engine::scalars::Scalars;

use super::session::{label_fields, text_at};

struct Label {
    text: String,
    /// the chunk the label names, for the caller's filter.
    ordinal: usize,
    /// what one line stands for: the first label, so its twins collapse.
    key: String,
    /// the rank column has a value for the chunk.
    ranked: bool,
    /// `u32::MAX - rank`, so higher sorts first; 0 when unranked.
    popularity: u32,
}

/// (ranked, tier, popularity in name tiers, score, -len): sorted descending.
type Rank = (bool, u8, u32, u32, i64);

pub struct Labels {
    /// in the index's order: a hit's id is a position here.
    items: Vec<Label>,
    index: Index,
}

impl Labels {
    /// Labels from the stored text in chunk order: the first line of each
    /// chunk, one label per distinct line. A label's ordinal is its position
    /// in `texts`, so the caller passes them in the order of the chunks.
    pub fn new<'a>(texts: impl Iterator<Item = &'a str>) -> Self {
        let mut seen = HashSet::new();
        let items = texts
            .enumerate()
            .filter_map(|(i, t)| t.lines().next().map(|l| (i, l.trim())))
            .filter(|(_, l)| !l.is_empty() && seen.insert(l.to_string()))
            .map(|(i, l)| {
                (
                    fold(l),
                    Label {
                        text: l.to_string(),
                        ordinal: i,
                        key: l.to_string(),
                        ranked: true,
                        popularity: 0,
                    },
                )
            })
            .collect();
        Self::build(items)
    }

    /// Labels from chunk_scalars: every label-role text column of every
    /// chunk (a name, then its translation), shown as `Name (Alias)`, with
    /// the rank-role column as popularity so the more played leads inside
    /// a tier and the unranked twins (an art series) come after.
    pub fn from_scalars(s: &Scalars) -> Self {
        let fields = label_fields(&s.schema);
        let rank = s
            .schema
            .iter()
            .find(|f| f.has_role("rank"))
            .and_then(|f| s.table.column(&f.name));
        let n = s.table.n_rows;
        let mut items = Vec::with_capacity(n * fields.len().max(1));
        for i in 0..n {
            let names: Vec<&str> = fields
                .iter()
                .filter_map(|f| text_at(s, &f.name, i))
                .collect();
            let Some(first) = names.first() else {
                continue;
            };
            let shown = match names.get(1) {
                Some(alias) => format!("{first} ({alias})"),
                None => first.to_string(),
            };
            let popularity = rank
                .filter(|c| c.present(i))
                .and_then(|c| match &c.data {
                    urna_format::sections::ColumnData::Int(v) => v.get(i).copied(),
                    _ => None,
                })
                .map_or(0, |r| u32::MAX.saturating_sub(r.max(0) as u32));
            for name in &names {
                items.push((
                    fold(name),
                    Label {
                        text: shown.clone(),
                        ordinal: i,
                        key: first.to_string(),
                        ranked: popularity > 0,
                        popularity,
                    },
                ));
            }
        }
        Self::build(items)
    }

    fn build(mut items: Vec<(String, Label)>) -> Self {
        items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.ordinal.cmp(&b.1.ordinal)));
        // sorted just above, so the index cannot refuse it.
        let index = Index::from_sorted(items.iter().map(|(f, l)| (f.as_str(), l.ordinal)))
            .unwrap_or_default();
        Self {
            items: items.into_iter().map(|(_, l)| l).collect(),
            index,
        }
    }

    #[cfg(test)]
    pub fn suggest(&self, query: &str, k: usize) -> Vec<String> {
        self.suggest_where(query, k, &|_| true)
    }

    /// The best `k` labels for `query` among the chunks `keep` accepts.
    pub fn suggest_where(
        &self,
        query: &str,
        k: usize,
        keep: &dyn Fn(usize) -> bool,
    ) -> Vec<String> {
        if k == 0 {
            return Vec::new();
        }
        let rank = |h: &urna_engine::labels::Hit| {
            let l = self.items.get(h.id)?;
            let popularity = if h.tier > 0 { l.popularity } else { 0 };
            let len = i64::try_from(h.len).unwrap_or(i64::MAX);
            Some(((l.ranked, h.tier, popularity, h.score, -len), l))
        };
        let exact = self.index.exact(query, keep);
        let mut scored: Vec<(Rank, &Label)> = exact.iter().filter_map(rank).collect();
        // a ranked label that q is a prefix of, or of one of its words,
        // outranks every subsequence match: with k of them the scan over
        // all the labels has nothing to add.
        let mut names: HashSet<&str> = HashSet::new();
        let enough = scored
            .iter()
            .filter(|(r, l)| r.0 && r.1 > 0 && names.insert(l.key.as_str()))
            .count()
            >= k;
        if !enough {
            let seen: HashSet<usize> = exact.iter().map(|h| h.id).collect();
            let fuzzy = self.index.fuzzy(query, keep);
            scored.extend(
                fuzzy
                    .iter()
                    .filter(|h| !seen.contains(&h.id))
                    .filter_map(rank),
            );
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.ordinal.cmp(&b.1.ordinal)));
        let mut seen: HashSet<&str> = HashSet::new();
        scored
            .into_iter()
            .filter(|(_, l)| seen.insert(l.key.as_str()))
            .take(k)
            .map(|(_, l)| l.text.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_label_beats_prefix_beats_fuzzy_and_accents_fold() {
        let texts = [
            "Lightning Bolt (Raio)\n{R}",
            "Lightning\nx",
            "Relâmpago Vivo\ny",
            "Bolt of Light\nz",
        ];
        let l = Labels::new(texts.into_iter());
        assert_eq!(l.suggest("lightning", 1), vec!["Lightning"]);
        assert_eq!(l.suggest("relampago", 1), vec!["Relâmpago Vivo"]);
        assert_eq!(l.suggest("raio", 1), vec!["Lightning Bolt (Raio)"]);
        assert_eq!(l.suggest("bolt light", 1), vec!["Bolt of Light"]);
        assert_eq!(
            l.suggest("bolt", 2),
            vec!["Bolt of Light", "Lightning Bolt (Raio)"]
        );
        assert_eq!(tier("lightning bolt (raio)", "raio"), 3);
        assert_eq!(tier("lightning bolt (raio)", "lightning"), 2);
        assert_eq!(tier("lightning bolt (raio)", "lightning bolt"), 3);
        assert_eq!(
            tier("lightning bolt // lightning bolt", "lightning bolt"),
            2
        );
        assert!(l.suggest("", 5).is_empty());
        assert_eq!(
            l.suggest_where("lightning", 5, &|o| o != 1),
            vec!["Lightning Bolt (Raio)"]
        );
    }

    #[test]
    fn scalars_offer_both_names_once_with_the_played_one_first() {
        use urna_format::manifest::schema::SchemaField;
        use urna_format::sections::{Column, ColumnData, ScalarTable};
        let f = |name: &str, roles: &[&str]| SchemaField {
            name: name.into(),
            kind: if name == "rank" {
                "int".into()
            } else {
                "text".into()
            },
            roles: roles.iter().map(|r| r.to_string()).collect(),
            ..Default::default()
        };
        let names = [
            "Lightning Bolt",
            "Lightning Bolt",
            "Lightning Helix",
            "Lightning Greaves",
            "Lightning",
        ];
        let pt = ["Raio", "", "Hélice de Raios", "", ""];
        let rank = [10, 0, 300, 1, 0];
        let s = Scalars {
            schema: vec![
                f("name", &["label"]),
                f("pt_name", &["label"]),
                f("rank", &["rank"]),
            ],
            table: ScalarTable {
                n_rows: 5,
                columns: vec![
                    Column {
                        name: "name".into(),
                        validity: None,
                        data: ColumnData::Text(names.iter().map(|s| s.to_string()).collect()),
                    },
                    Column {
                        name: "pt_name".into(),
                        validity: Some(pt.iter().map(|p| !p.is_empty()).collect()),
                        data: ColumnData::Text(pt.iter().map(|s| s.to_string()).collect()),
                    },
                    Column {
                        name: "rank".into(),
                        validity: Some(rank.iter().map(|r| *r > 0).collect()),
                        data: ColumnData::Int(rank.to_vec()),
                    },
                ],
            },
        };
        let l = Labels::from_scalars(&s);
        assert_eq!(
            l.suggest("lightn", 5),
            vec![
                "Lightning Greaves",
                "Lightning Bolt (Raio)",
                "Lightning Helix (Hélice de Raios)",
                "Lightning",
            ]
        );
        assert_eq!(l.suggest("raio", 1), vec!["Lightning Bolt (Raio)"]);
        assert_eq!(
            l.suggest("helice", 1),
            vec!["Lightning Helix (Hélice de Raios)"]
        );
        assert_eq!(
            l.suggest_where("lightning bolt", 5, &|o| o != 0 && o != 1),
            Vec::<String>::new()
        );
    }
}
