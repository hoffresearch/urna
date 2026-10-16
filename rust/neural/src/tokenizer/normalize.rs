//! BERT's normalizer, as the `tokenizer.json` declares it: clean_text
//! (controls dropped, every whitespace a plain space), spaces around CJK
//! ideographs, accents stripped (NFD then the nonspacing marks removed,
//! on by default when lowercasing), lowercase.

use serde::Deserialize;

use super::unicode;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum NormalizerSpec {
    BertNormalizer {
        #[serde(default = "yes")]
        clean_text: bool,
        #[serde(default = "yes")]
        handle_chinese_chars: bool,
        #[serde(default)]
        strip_accents: Option<bool>,
        #[serde(default = "yes")]
        lowercase: bool,
    },
}

fn yes() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BertNormalizer {
    pub clean_text: bool,
    pub handle_chinese_chars: bool,
    pub strip_accents: bool,
    pub lowercase: bool,
}

impl From<NormalizerSpec> for BertNormalizer {
    fn from(spec: NormalizerSpec) -> Self {
        let NormalizerSpec::BertNormalizer {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        } = spec;
        Self {
            clean_text,
            handle_chinese_chars,
            strip_accents: strip_accents.unwrap_or(lowercase),
            lowercase,
        }
    }
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && unicode::is_other(c)
}

fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

impl BertNormalizer {
    pub fn normalize(&self, text: &str) -> String {
        let mut s: String = if self.clean_text {
            text.chars()
                .filter(|&c| !(c == '\0' || c == '\u{FFFD}' || is_control(c)))
                .map(|c| if is_whitespace(c) { ' ' } else { c })
                .collect()
        } else {
            text.to_string()
        };
        if self.handle_chinese_chars {
            let mut spaced = String::with_capacity(s.len() + 8);
            for c in s.chars() {
                if is_chinese_char(c) {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            s = spaced;
        }
        if self.strip_accents {
            s = unicode::nfd(&s)
                .chars()
                .filter(|&c| !unicode::is_mark_nonspacing(c))
                .collect();
        }
        if self.lowercase {
            s = s.chars().flat_map(char::to_lowercase).collect();
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_spaces_cjk_accents_and_case_like_bert() {
        let n: BertNormalizer = serde_json::from_str::<NormalizerSpec>(
            r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#,
        )
        .expect("spec")
        .into();
        assert!(n.strip_accents);
        assert_eq!(n.normalize("Relâmpago\tVivo\u{0}"), "relampago vivo");
        assert_eq!(n.normalize("A中B"), "a 中 b");
        assert_eq!(
            n.normalize("İ"),
            "i\u{307}"
                .chars()
                .filter(|c| !unicode::is_mark_nonspacing(*c))
                .collect::<String>()
                .to_lowercase()
                .replace('\u{307}', "")
        );
        let keep: BertNormalizer = NormalizerSpec::BertNormalizer {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: Some(false),
            lowercase: false,
        }
        .into();
        assert_eq!(keep.normalize("Relâmpago"), "Relâmpago");
    }
}
