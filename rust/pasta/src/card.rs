//! A card as the fields file stores it: the dataset's own columns plus what
//! Scryfall's oracle bulk adds for decks (colors, cmc, legality, popularity).

use std::collections::BTreeMap;

use serde::Deserialize;

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
    pub oracle_id: String,
    pub name: String,
    pub pt_name: Option<String>,
    pub mana_cost: String,
    pub type_line: String,
    pub rarity: String,
    pub set_code: String,
    pub oracle_text: String,
    pub kind: Kind,
    #[serde(default)]
    pub colors: Vec<String>,
    #[serde(default)]
    pub color_identity: Vec<String>,
    pub cmc: Option<f32>,
    pub power: Option<String>,
    pub toughness: Option<String>,
    pub loyalty: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    pub layout: Option<String>,
    /// format -> `legal`, `restricted` or `banned`; formats where the card
    /// is not legal are left out.
    #[serde(default)]
    pub legal: BTreeMap<String, String>,
    /// EDHREC's rank, 1 the most played; `None` when EDHREC has no data.
    pub edhrec_rank: Option<u32>,
    #[serde(default)]
    pub games: Vec<String>,
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
}

#[derive(Deserialize)]
pub(crate) struct FieldsFile {
    pub format: String,
    pub content_hash: String,
    pub cards: Vec<Card>,
}
