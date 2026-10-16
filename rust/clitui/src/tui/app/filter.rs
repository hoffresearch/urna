//! What a deck builder narrows by, typed after `|` in the ask tab:
//! `format=commander id=RG cmc<=3 type=creature all`, in any order. Every
//! field is optional; the default keeps the cards a paper deck can hold
//! and drops nothing else.

use super::fields::Card;

#[derive(Clone, Debug, Default, PartialEq)]
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

    /// The text before `|` and the filter after it.
    pub fn split(input: &str) -> (&str, Self) {
        let (q, spec) = input.split_once('|').unwrap_or((input, ""));
        (q.trim(), Self::parse(spec))
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
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_field_in_any_order_and_splits_the_input() {
        let f = Filter::parse("type=Dragon all cmc<=3 id=rg format=Commander");
        assert!(f.include_unplayable);
        assert_eq!(f.format.as_deref(), Some("commander"));
        assert_eq!(f.identity_within, Some(vec!["R".into(), "G".into()]));
        assert_eq!(f.max_cmc, Some(3.0));
        assert_eq!(f.type_contains.as_deref(), Some("Dragon"));
        assert!(!f.is_default());
        assert!(Filter::parse("").is_default());
        let (q, f) = Filter::split(" draw three cards | format=pauper ");
        assert_eq!(q, "draw three cards");
        assert_eq!(f.format.as_deref(), Some("pauper"));
        assert_eq!(Filter::split("bolt"), ("bolt", Filter::default()));
    }
}
