//! The search syntax people who play already know, over the schema's
//! prefixes: `t:creature c:r pow>=3 o:"draw a card" bolt`. A `prefix:value`
//! or `prefix=value` term is equality (or membership on a list, substring
//! on a text), `prefix>=3`, `>`, `<`, `<=`, `!=` are ranges on a numeric
//! column, `-term` negates, `o:` words go to the lexical search over the
//! stored text, and what has no prefix is the free text of the query. The
//! prefixes come from the manifest schema, never from code, so another
//! corpus brings its own.

use urna_format::manifest::schema::SchemaField;

use super::{Filter, Predicate};
use crate::error::RuntimeError;

/// What a query line said: the predicates and the text left for search.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedQuery {
    pub filter: Filter,
    /// the words without a prefix plus the `o:` words, for the vector and
    /// lexical search.
    pub text: String,
    /// the raw terms that became predicates, for the explain line.
    pub terms: Vec<String>,
}

/// Parse `line` with the schema's prefixes. The `o` prefix (oracle text)
/// is reserved for the stored text and needs no schema entry.
pub fn parse(line: &str, schema: &[SchemaField]) -> Result<ParsedQuery, RuntimeError> {
    let mut out = ParsedQuery::default();
    let mut free: Vec<String> = Vec::new();
    for raw in tokens(line) {
        let (negate, term) = match raw.strip_prefix('-') {
            Some(rest) if !rest.is_empty() => (true, rest),
            _ => (false, raw.as_str()),
        };
        let Some((prefix, op, value)) = split_term(term) else {
            if negate && term.starts_with(|c: char| c.is_alphabetic() || c == '"') {
                return Err(cannot_exclude(&raw));
            }
            // `-1/-1` is text, not a negation.
            free.push(unquote(&raw));
            continue;
        };
        if prefix.eq_ignore_ascii_case("o") {
            if negate {
                return Err(cannot_exclude(&raw));
            }
            free.push(unquote(value));
            continue;
        }
        let field = schema
            .iter()
            .find(|f| {
                f.prefix
                    .as_deref()
                    .is_some_and(|p| p.eq_ignore_ascii_case(prefix))
            })
            .ok_or_else(|| RuntimeError::Filter(format!("unknown search prefix {prefix:?}")))?;
        let value = unquote(value);
        if value.is_empty() {
            return Err(RuntimeError::Filter(format!("{raw}: the value is empty")));
        }
        let p = predicate(field, op, &value)?;
        out.filter.predicates.push(if negate {
            Predicate::Not(Box::new(p))
        } else {
            p
        });
        out.terms.push(raw.clone());
    }
    out.text = free.join(" ");
    Ok(out)
}

fn predicate(field: &SchemaField, op: &str, value: &str) -> Result<Predicate, RuntimeError> {
    let column = field.name.clone();
    let numeric = matches!(field.kind.as_str(), "int" | "real")
        || (field.kind == "enum" && matches!(op, ">=" | ">" | "<=" | "<"));
    if op != ":" && op != "=" && op != "!=" && !numeric {
        return Err(RuntimeError::Filter(format!(
            "{} takes `:` or `=`, not {op}",
            field.name
        )));
    }
    Ok(match (field.kind.as_str(), op) {
        ("enum_map", ":" | "=") => Predicate::Eq {
            column: format!("{column}.{}", value.to_lowercase()),
            value: "legal".into(),
        },
        ("enum_map", "!=") => Predicate::Not(Box::new(Predicate::Eq {
            column: format!("{column}.{}", value.to_lowercase()),
            value: "legal".into(),
        })),
        ("enum_list", ":" | "=") => Predicate::Has {
            column,
            value: value.into(),
        },
        ("enum_list", "!=") => Predicate::Not(Box::new(Predicate::Has {
            column,
            value: value.into(),
        })),
        ("text", ":") => Predicate::Contains {
            column,
            value: value.into(),
        },
        ("text", "=") => Predicate::Eq {
            column,
            value: value.into(),
        },
        ("text", "!=") => Predicate::Not(Box::new(Predicate::Eq {
            column,
            value: value.into(),
        })),
        (_, ":" | "=") if !numeric => Predicate::Eq {
            column,
            value: value.into(),
        },
        (_, "!=") if !numeric => Predicate::Not(Box::new(Predicate::Eq {
            column,
            value: value.into(),
        })),
        _ => {
            let x: f64 = value.parse().map_err(|_| {
                RuntimeError::Filter(format!("{value:?} is not a number for {}", field.name))
            })?;
            match op {
                ":" | "=" => Predicate::Range {
                    column,
                    min: Some((x, true)),
                    max: Some((x, true)),
                },
                "!=" => Predicate::Not(Box::new(Predicate::Range {
                    column,
                    min: Some((x, true)),
                    max: Some((x, true)),
                })),
                ">=" => Predicate::Range {
                    column,
                    min: Some((x, true)),
                    max: None,
                },
                ">" => Predicate::Range {
                    column,
                    min: Some((x, false)),
                    max: None,
                },
                "<=" => Predicate::Range {
                    column,
                    min: None,
                    max: Some((x, true)),
                },
                "<" => Predicate::Range {
                    column,
                    min: None,
                    max: Some((x, false)),
                },
                other => return Err(RuntimeError::Filter(format!("unknown operator {other:?}"))),
            }
        }
    })
}

/// the search runs over the stored chunks and the scalar columns; there
/// is no text column to exclude a free word or an `o:` phrase from.
fn cannot_exclude(raw: &str) -> RuntimeError {
    RuntimeError::Filter(format!(
        "{raw}: `-` excludes on a prefixed column, not on free text or o:"
    ))
}

/// `prefix`, operator and value of a term, split at its first operator
/// (`>=` before `>` at the same place), so a value may hold operators.
fn split_term(term: &str) -> Option<(&str, &str, &str)> {
    let (i, op) = term.char_indices().find_map(|(i, _)| {
        [">=", "<=", "!=", ":", "=", ">", "<"]
            .into_iter()
            .find(|op| term[i..].starts_with(op))
            .map(|op| (i, op))
    })?;
    let (prefix, rest) = term.split_at(i);
    let value = &rest[op.len()..];
    let valid = !prefix.is_empty()
        && prefix
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !value.is_empty();
    valid.then_some((prefix, op, value))
}

fn unquote(s: &str) -> String {
    s.strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .unwrap_or(s)
        .to_string()
}

/// whitespace-separated terms, with double quotes keeping spaces.
fn tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                cur.push(c);
            }
            c if c.is_whitespace() && !quoted => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Vec<SchemaField> {
        let f = |name: &str, kind: &str, prefix: &str| SchemaField {
            name: name.into(),
            kind: kind.into(),
            prefix: Some(prefix.into()),
            ..Default::default()
        };
        vec![
            f("type_line", "text", "t"),
            f("colors", "enum_list", "c"),
            f("power", "enum", "pow"),
            f("cmc", "real", "cmc"),
            f("rarity", "enum", "r"),
            f("legal", "enum_map", "f"),
            f("keywords", "enum_list", "kw"),
        ]
    }

    #[test]
    fn parses_prefixes_operators_quotes_negation_and_free_text() {
        let q = parse(
            r#"t:creature c:rg pow>=3 -r:common f:pauper o:"draw a card" bolt cmc<2"#,
            &schema(),
        )
        .expect("parses");
        assert_eq!(q.text, "draw a card bolt");
        assert_eq!(q.filter.predicates.len(), 6);
        assert_eq!(
            q.filter.predicates[0],
            Predicate::Contains {
                column: "type_line".into(),
                value: "creature".into()
            }
        );
        assert_eq!(
            q.filter.predicates[1],
            Predicate::Has {
                column: "colors".into(),
                value: "rg".into()
            }
        );
        assert_eq!(
            q.filter.predicates[2],
            Predicate::Range {
                column: "power".into(),
                min: Some((3.0, true)),
                max: None
            }
        );
        assert!(matches!(q.filter.predicates[3], Predicate::Not(_)));
        assert_eq!(
            q.filter.predicates[4],
            Predicate::Eq {
                column: "legal.pauper".into(),
                value: "legal".into()
            }
        );
        assert_eq!(
            q.filter.predicates[5],
            Predicate::Range {
                column: "cmc".into(),
                min: None,
                max: Some((2.0, false))
            }
        );
        assert!(parse("zz:1", &schema()).is_err());
        assert!(parse("t>3", &schema()).is_err());
        assert_eq!(
            parse("lightning bolt", &schema()).expect("free").text,
            "lightning bolt"
        );
    }

    #[test]
    fn a_list_word_stays_whole_and_not_equal_on_an_enum_is_not_a_range() {
        let one = |line: &str| parse(line, &schema()).expect("parses").filter.predicates[0].clone();
        assert_eq!(
            one("kw:haste"),
            Predicate::Has {
                column: "keywords".into(),
                value: "haste".into()
            }
        );
        let not_eq = |column: &str, value: &str| {
            Predicate::Not(Box::new(Predicate::Eq {
                column: column.into(),
                value: value.into(),
            }))
        };
        assert_eq!(one("r!=common"), not_eq("rarity", "common"));
        assert_eq!(one("pow!=*"), not_eq("power", "*"));
    }

    #[test]
    fn a_negation_without_a_column_is_an_error_not_a_dropped_sign() {
        assert!(parse("t:creature -o:flying", &schema()).is_err());
        assert!(parse("-bolt", &schema()).is_err());
        assert!(parse(r#"-"lightning bolt""#, &schema()).is_err());
        assert_eq!(
            parse("-1/-1 counter", &schema()).expect("text").text,
            "-1/-1 counter"
        );
    }

    #[test]
    fn an_empty_quoted_value_is_an_error_not_every_row() {
        for line in [r#"c:"""#, r#"t:"""#, r#"-c:"""#, r#"cmc>="""#] {
            let e = parse(line, &schema()).expect_err(line);
            assert!(e.to_string().contains("the value is empty"), "{line}: {e}");
        }
    }

    #[test]
    fn the_first_operator_splits_the_term() {
        let q = parse(r#"o:"x >= 3" cmc>=2"#, &schema()).expect("parses");
        assert_eq!(q.text, "x >= 3");
        assert_eq!(q.filter.predicates.len(), 1);
    }
}
