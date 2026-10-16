//! What a deck builder narrows by. Every field is optional; the default
//! keeps the cards a paper deck can hold and drops nothing else.

use crate::card::Card;

#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// Keep art-series, tokens, emblems and Alchemy cards too.
    pub include_unplayable: bool,
    /// `commander`, `modern`, `pauper`...: legal or restricted there.
    pub format: Option<String>,
    /// The card's color identity fits inside these (`W U B R G`); colorless
    /// cards always fit. Commander's rule.
    pub identity_within: Option<Vec<String>>,
    pub max_cmc: Option<f32>,
    /// Case-insensitive substring of the type line: `creature`, `instant`.
    pub type_contains: Option<String>,
}

impl Filter {
    /// `format=commander id=RG cmc<=3 type=creature all`, in any order.
    pub fn parse(spec: &str) -> Self {
        let mut f = Self::default();
        for part in spec.split_whitespace() {
            if part == "all" {
                f.include_unplayable = true;
            } else if let Some(v) = part.strip_prefix("format=") {
                f.format = Some(v.to_lowercase());
            } else if let Some(v) = part.strip_prefix("id=") {
                f.identity_within = Some(
                    v.chars()
                        .map(|c| c.to_ascii_uppercase().to_string())
                        .collect(),
                );
            } else if let Some(v) = part.strip_prefix("cmc<=") {
                f.max_cmc = v.parse().ok();
            } else if let Some(v) = part.strip_prefix("type=") {
                f.type_contains = Some(v.to_string());
            }
        }
        f
    }

    pub fn keeps(&self, c: &Card) -> bool {
        if !self.include_unplayable && !c.playable() {
            return false;
        }
        if let Some(f) = &self.format
            && !c.legal_in(f)
        {
            return false;
        }
        if let Some(allowed) = &self.identity_within
            && !c.color_identity.iter().all(|x| allowed.contains(x))
        {
            return false;
        }
        if let Some(max) = self.max_cmc
            && c.cmc.is_none_or(|v| v > max)
        {
            return false;
        }
        if let Some(t) = &self.type_contains
            && !c.type_line.to_lowercase().contains(&t.to_lowercase())
        {
            return false;
        }
        true
    }

    pub fn is_default(&self) -> bool {
        !self.include_unplayable
            && self.format.is_none()
            && self.identity_within.is_none()
            && self.max_cmc.is_none()
            && self.type_contains.is_none()
    }
}
