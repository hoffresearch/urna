//! Fitting text into columns: keep the end of a path, wrap prose, and
//! show the home directory as `~`.

/// Every `$HOME` prefix in `s` as `~` (display only; paths stay real).
pub fn home(s: &str) -> String {
    match std::env::var("HOME") {
        Ok(h) if h.len() > 1 => s.replace(&h, "~"),
        _ => s.to_string(),
    }
}

/// `s` cut to `w` columns keeping its end (`…/share/urna/python`), which
/// is the informative half of a path or a hash.
pub fn tail(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n <= w {
        return s.to_string();
    }
    if w == 0 {
        return String::new();
    }
    let keep: String = s.chars().skip(n - (w - 1)).collect();
    format!("…{keep}")
}

/// Greedy word wrap into lines of at most `w` columns (a word longer than
/// the line is cut, not dropped).
pub fn wrap(s: &str, w: usize) -> Vec<String> {
    let w = w.max(1);
    let mut lines = vec![String::new()];
    for word in s.split_whitespace() {
        let cur = lines.last_mut().map(|l| l.chars().count()).unwrap_or(0);
        let need = word.chars().count() + usize::from(cur > 0);
        if cur > 0 && cur + need > w {
            lines.push(String::new());
        }
        if let Some(l) = lines.last_mut() {
            if !l.is_empty() {
                l.push(' ');
            }
            l.push_str(&word.chars().take(w).collect::<String>());
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_breaks_on_words() {
        assert_eq!(
            wrap("download thirty megabytes now", 12),
            vec!["download", "thirty", "megabytes", "now"]
        );
        assert_eq!(wrap("a b c", 10), vec!["a b c"]);
        assert_eq!(wrap("", 5), vec![""]);
    }

    #[test]
    fn tail_keeps_the_end_of_long_values() {
        assert_eq!(tail("/a/b/c/python", 7), "…python");
        assert_eq!(tail("short", 10), "short");
        assert_eq!(tail("abc", 0), "");
    }
}
