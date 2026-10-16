//! The card fields a corpus can carry beside the `.urna`, provisional until
//! the file carries them itself (#208): one row per chunk ordinal, with the
//! columns the explorer reads (the file carries more, left to #208): names,
//! kind, identity, cmc, stats, legality and popularity. The file is zstd json bound to the
//! corpus by its `content_hash`, and the explorer finds it as
//! `<corpus stem>.fields.json.zst` next to the corpus or at `URNA_FIELDS`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const FORMAT: &str = "pasta-fields/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Card,
    Token,
    Emblem,
    ArtSeries,
    /// Arena's rebalanced `A-` cards: digital only, no paper deck holds them.
    Alchemy,
    /// Legal in no format: minigames, Un-set oddities, playtest cards.
    Unsanctioned,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Card {
    pub ordinal: usize,
    pub name: String,
    pub pt_name: Option<String>,
    pub type_line: String,
    pub kind: Kind,
    #[serde(default)]
    pub color_identity: Vec<String>,
    pub cmc: Option<f32>,
    pub power: Option<String>,
    pub toughness: Option<String>,
    pub loyalty: Option<String>,
    /// format -> `legal`, `restricted` or `banned`; formats where the card
    /// is not legal are left out.
    #[serde(default)]
    pub legal: BTreeMap<String, String>,
    /// EDHREC's rank, 1 the most played; `None` when EDHREC has no data.
    pub edhrec_rank: Option<u32>,
}

impl Card {
    /// Cards a paper deck can hold.
    pub fn playable(&self) -> bool {
        self.kind == Kind::Card
    }

    pub fn legal_in(&self, format: &str) -> bool {
        matches!(
            self.legal.get(format).map(String::as_str),
            Some("legal" | "restricted")
        )
    }

    /// The formats the card is legal or restricted in, in the file's order.
    pub fn formats(&self) -> impl Iterator<Item = &str> {
        self.legal
            .iter()
            .filter(|(_, s)| matches!(s.as_str(), "legal" | "restricted"))
            .map(|(f, _)| f.as_str())
    }
}

#[derive(Deserialize)]
struct FieldsFile {
    format: String,
    content_hash: String,
    cards: Vec<Card>,
}

/// Where the fields of `corpus` would be: `URNA_FIELDS` when set, else
/// `<stem>.fields.json.zst` beside it. `None` when neither exists.
pub fn locate(corpus: &Path) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("URNA_FIELDS") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let stem = corpus.file_stem()?.to_str()?;
    let p = corpus.with_file_name(format!("{stem}.fields.json.zst"));
    p.is_file().then_some(p)
}

/// Reads and validates the fields file: its format, that it belongs to the
/// corpus with `content_hash`, and that it holds exactly `n` cards in
/// ordinal order.
pub fn read(path: &Path, content_hash: &str, n: usize) -> Result<Vec<Card>> {
    let raw = std::fs::read(path).with_context(|| format!("fields {}", path.display()))?;
    let json = zstd::decode_all(raw.as_slice()).context("fields: zstd")?;
    let doc: FieldsFile = serde_json::from_slice(&json).context("fields: json")?;
    if doc.format != FORMAT {
        bail!("fields: unknown format {}, expected {FORMAT}", doc.format);
    }
    if doc.content_hash != content_hash {
        bail!(
            "the fields file belongs to {}, the corpus is {content_hash}",
            doc.content_hash
        );
    }
    if doc.cards.len() != n || doc.cards.iter().enumerate().any(|(i, c)| c.ordinal != i) {
        bail!(
            "fields: expected ordinals 0..{n} in order, got {} cards",
            doc.cards.len()
        );
    }
    Ok(doc.cards)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(ordinal: usize, kind: &str) -> String {
        format!(
            r#"{{"ordinal":{ordinal},"oracle_id":"o{ordinal}","name":"N{ordinal}","pt_name":null,
            "mana_cost":"{{R}}","type_line":"Instant","rarity":"common","set_code":"lea",
            "oracle_text":"","kind":"{kind}","cmc":1.0,"power":null,"toughness":null,
            "loyalty":null,"layout":"normal","legal":{{"pauper":"legal","modern":"banned"}},
            "edhrec_rank":{ordinal}}}"#
        )
    }

    fn file(hash: &str, cards: &[String]) -> Vec<u8> {
        let json = format!(
            r#"{{"format":"pasta-fields/1","content_hash":"{hash}","cards":[{}]}}"#,
            cards.join(",")
        );
        zstd::encode_all(json.as_bytes(), 1).expect("zstd")
    }

    #[test]
    fn reads_a_file_bound_to_the_corpus_and_refuses_the_rest() {
        let dir = tempfile::tempdir().expect("tmp");
        let p = dir.path().join("c.fields.json.zst");
        std::fs::write(
            &p,
            file("sha256:c", &[card(0, "card"), card(1, "art_series")]),
        )
        .expect("write");
        let cards = read(&p, "sha256:c", 2).expect("reads");
        assert!(cards[0].playable() && !cards[1].playable());
        assert!(cards[0].legal_in("pauper") && !cards[0].legal_in("modern"));
        assert_eq!(cards[0].formats().collect::<Vec<_>>(), vec!["pauper"]);
        assert!(read(&p, "sha256:other", 2).is_err());
        assert!(read(&p, "sha256:c", 3).is_err());
        std::fs::write(&p, file("sha256:c", &[card(1, "card")])).expect("write");
        assert!(read(&p, "sha256:c", 1).is_err());
    }

    #[test]
    fn locates_the_sidecar_by_the_corpus_stem() {
        let dir = tempfile::tempdir().expect("tmp");
        let corpus = dir.path().join("mtgdataset.urna");
        assert_eq!(locate(&corpus), None);
        let side = dir.path().join("mtgdataset.fields.json.zst");
        std::fs::write(&side, b"x").expect("write");
        assert_eq!(locate(&corpus), Some(side));
    }
}
