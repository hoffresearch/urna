//! BERT's pre-tokenizer: split on whitespace (dropped), then isolate every
//! punctuation character (ASCII punctuation as `is_ascii_punctuation`
//! counts it, plus the Unicode P* categories) as a word of its own.

use super::unicode;

fn is_bert_punc(c: char) -> bool {
    c.is_ascii_punctuation() || unicode::is_punctuation(c)
}

/// The words of `text`, in order, as the WordPiece model sees them.
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for piece in text.split(char::is_whitespace) {
        if piece.is_empty() {
            continue;
        }
        let mut start = 0;
        for (i, c) in piece.char_indices() {
            if is_bert_punc(c) {
                if start < i {
                    out.push(&piece[start..i]);
                }
                out.push(&piece[i..i + c.len_utf8()]);
                start = i + c.len_utf8();
            }
        }
        if start < piece.len() {
            out.push(&piece[start..]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_the_reference() {
        assert_eq!(
            pre_tokenize("Hey friend!     How are you?!?"),
            vec!["Hey", "friend", "!", "How", "are", "you", "?", "!", "?"]
        );
        assert_eq!(pre_tokenize("a$b «c»"), vec!["a", "$", "b", "«", "c", "»"]);
        assert!(pre_tokenize("   ").is_empty());
    }
}
