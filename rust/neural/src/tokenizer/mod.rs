//! Tokenizers read from each model's `tokenizer.json`, the Hugging Face
//! layout: the added tokens are matched first on the raw text (leftmost,
//! longest), each remaining piece is normalized, pre-tokenized into words
//! and cut by the model. WordPiece with BERT's normalizer and
//! pre-tokenizer lands with the potion; other models bring theirs.

mod normalize;
mod pretok;
mod unicode;
mod wordpiece;

use std::path::Path;

use serde::Deserialize;

use crate::{InferError, Result};

pub use normalize::BertNormalizer;
pub use wordpiece::WordPiece;

#[derive(Clone, Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct PreTokenizerSpec {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Clone, Debug, Deserialize)]
struct File {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    normalizer: Option<normalize::NormalizerSpec>,
    pre_tokenizer: Option<PreTokenizerSpec>,
    model: wordpiece::ModelSpec,
}

/// A loaded tokenizer: `encode` gives the ids a text takes, with no
/// special tokens added (the potion's `add_special_tokens=False`).
#[derive(Clone, Debug)]
pub struct Tokenizer {
    normalizer: BertNormalizer,
    model: WordPiece,
    /// added tokens matched on the raw text, longest first.
    added: Vec<(String, u32)>,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| InferError::Tokenizer(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let file: File = serde_json::from_str(text)?;
        let normalizer = file
            .normalizer
            .ok_or_else(|| InferError::Tokenizer("no normalizer; BertNormalizer expected".into()))?
            .into();
        match file.pre_tokenizer.as_ref().map(|p| p.kind.as_str()) {
            Some("BertPreTokenizer") => {}
            other => {
                return Err(InferError::Tokenizer(format!(
                    "pre_tokenizer {other:?} is not BertPreTokenizer"
                )));
            }
        }
        let model = WordPiece::new(file.model)?;
        let mut added: Vec<(String, u32)> = Vec::new();
        for t in file.added_tokens {
            if t.normalized || t.single_word || t.lstrip || t.rstrip {
                return Err(InferError::Tokenizer(format!(
                    "added token {:?} needs matching options this tokenizer does not carry",
                    t.content
                )));
            }
            if t.content.is_empty() {
                continue;
            }
            let _ = t.special;
            added.push((t.content, t.id));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        Ok(Self {
            normalizer,
            model,
            added,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(c, _)| c == token)
            .map(|&(_, id)| id)
            .or_else(|| self.model.id_of(token))
    }

    pub fn unk_id(&self) -> u32 {
        self.model.unk_id()
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// The ids of `text`, no special tokens added.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // leftmost match among the added tokens, longest first on a tie.
            let hit = self
                .added
                .iter()
                .filter_map(|(c, id)| rest.find(c.as_str()).map(|at| (at, c.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            match hit {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let normalized = self.normalizer.normalize(text);
        for word in pretok::pre_tokenize(&normalized) {
            self.model.tokenize(word, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JSON: &str = r###"{"version":"1.0","added_tokens":[{"id":1,"content":"[UNK]","special":true},{"id":3,"content":"[SEP]","special":true}],
        "normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
        "pre_tokenizer":{"type":"BertPreTokenizer"},
        "model":{"type":"WordPiece","vocab":{"[UNK]":1,"[SEP]":3,"light":10,"##ning":11,"bolt":12,",":13,"raio":14},"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100}}"###;

    #[test]
    fn encodes_through_added_tokens_normalizer_pretokenizer_and_model() {
        let t = Tokenizer::from_json(JSON).expect("loads");
        assert_eq!(t.encode("Lightning Bolt, Ráio"), vec![10, 11, 12, 13, 14]);
        assert_eq!(t.encode("bolt[SEP]bolt [UNK]"), vec![12, 3, 12, 1]);
        assert_eq!(t.encode(""), Vec::<u32>::new());
        assert_eq!(t.encode("zzz"), vec![1]);
        assert_eq!(t.token_to_id("[UNK]"), Some(1));
        assert_eq!(t.unk_id(), 1);
    }
}
