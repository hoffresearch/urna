//! The WordPiece model: a word is cut greedily into the longest vocabulary
//! piece from the left, every piece after the first carrying the
//! continuation prefix; a word longer than the limit, or one with a part
//! no piece covers, is the unknown token whole.

use std::collections::HashMap;

use serde::Deserialize;

use crate::{InferError, Result};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ModelSpec {
    WordPiece {
        vocab: HashMap<String, u32>,
        unk_token: String,
        #[serde(default = "hashes")]
        continuing_subword_prefix: String,
        #[serde(default = "hundred")]
        max_input_chars_per_word: usize,
    },
}

fn hashes() -> String {
    "##".into()
}

fn hundred() -> usize {
    100
}

#[derive(Clone, Debug)]
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    unk: u32,
    prefix: String,
    max_chars: usize,
}

impl WordPiece {
    pub fn new(spec: ModelSpec) -> Result<Self> {
        let ModelSpec::WordPiece {
            vocab,
            unk_token,
            continuing_subword_prefix,
            max_input_chars_per_word,
        } = spec;
        let unk = *vocab.get(&unk_token).ok_or_else(|| {
            InferError::Tokenizer(format!("unk token {unk_token:?} not in vocab"))
        })?;
        Ok(Self {
            vocab,
            unk,
            prefix: continuing_subword_prefix,
            max_chars: max_input_chars_per_word,
        })
    }

    pub fn unk_id(&self) -> u32 {
        self.unk
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The ids of one word.
    pub fn tokenize(&self, word: &str, out: &mut Vec<u32>) {
        if word.chars().count() > self.max_chars {
            out.push(self.unk);
            return;
        }
        let mark = out.len();
        let mut start = 0;
        let mut buf = String::new();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                let piece = &word[start..end];
                let key: &str = if start > 0 {
                    buf.clear();
                    buf.push_str(&self.prefix);
                    buf.push_str(piece);
                    &buf
                } else {
                    piece
                };
                if let Some(&id) = self.vocab.get(key) {
                    found = Some(id);
                    break;
                }
                end -= piece.chars().last().map_or(1, char::len_utf8);
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(mark);
                    out.push(self.unk);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> WordPiece {
        let spec: ModelSpec = serde_json::from_str(
            r###"{"type":"WordPiece","vocab":{"[UNK]":1,"light":10,"##ning":11,"bolt":12,"##s":13},"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":6}"###,
        )
        .expect("spec");
        WordPiece::new(spec).expect("model")
    }

    #[test]
    fn cuts_greedily_and_falls_back_to_unk() {
        let m = model();
        let mut out = Vec::new();
        for w in ["lightning", "bolts", "zap", "lightningx"] {
            m.tokenize(w, &mut out);
        }
        assert_eq!(out, vec![1, 12, 13, 1, 1]);
        let mut out = Vec::new();
        m.tokenize("bolt", &mut out);
        assert_eq!(out, vec![12]);
        assert_eq!(m.unk_id(), 1);
    }
}
