//! unicode-aware token splitter used by both the index builder and
//! search path. Lowercase, split on non-alphanumeric, drop tokens of
//! length < 2.

/// Lowercase, split on non-alphanumerics, drop tokens of length < 2.
/// Unicode-aware: handles PT-BR accents (ã, ç, õ) by virtue of `char`
/// iteration. Simple but effective enough for fake-news / news corpora.
pub(crate) fn tokenize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() {
            for low in c.to_lowercase() {
                cur.push(low);
            }
        } else if !cur.is_empty() {
            if cur.chars().count() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.chars().count() >= 2 {
        out.push(cur);
    }
    out
}
