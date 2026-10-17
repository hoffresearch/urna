//! Predicates over chunk_scalars columns, resolved to a bitset. Equality
//! and range on numbers (int, real, and an enum with a numeric table),
//! equality on enum, text and bool, membership in a list, `and` between
//! them, and `not` on any. A predicate on a column the file lacks, or of a
//! kind that cannot take it, is a typed error, never an empty result.

use urna_format::sections::{Column, ColumnData, ScalarTable};

use super::Bitset;
use crate::error::RuntimeError;

#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    /// the value equals `value`: an enum entry, a text (case-insensitive,
    /// whole), a bool (`true`/`false`), or a number.
    Eq {
        column: String,
        value: String,
    },
    /// a text column contains `value`, case-insensitive.
    Contains {
        column: String,
        value: String,
    },
    /// a list column holds `value`; on a dictionary of single letters a
    /// longer value holds each of its letters (`rg`).
    Has {
        column: String,
        value: String,
    },
    /// a numeric column (or an enum with numbers) within the bounds; each
    /// bound is `(value, inclusive)`.
    Range {
        column: String,
        min: Option<(f64, bool)>,
        max: Option<(f64, bool)>,
    },
    /// the column has a value in the row.
    Present {
        column: String,
    },
    Not(Box<Predicate>),
    /// every predicate holds (an `and` inside a single term).
    All(Vec<Predicate>),
}

/// an `and` of predicates; empty keeps every chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Filter {
    pub predicates: Vec<Predicate>,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }

    pub fn and(mut self, p: Predicate) -> Self {
        self.predicates.push(p);
        self
    }

    /// The chunks every predicate keeps.
    pub fn bitset(&self, table: &ScalarTable, n: usize) -> Result<Bitset, RuntimeError> {
        let mut out = Bitset::full(n);
        for p in &self.predicates {
            out.and(&eval(p, table, n)?);
        }
        Ok(out)
    }
}

fn err(msg: String) -> RuntimeError {
    RuntimeError::Filter(msg)
}

fn column<'a>(table: &'a ScalarTable, name: &str) -> Result<&'a Column, RuntimeError> {
    table
        .column(name)
        .ok_or_else(|| err(format!("no column {name:?} in this corpus")))
}

fn eval(p: &Predicate, table: &ScalarTable, n: usize) -> Result<Bitset, RuntimeError> {
    Ok(match p {
        Predicate::Not(inner) => {
            let mut b = eval(inner, table, n)?;
            b.not();
            b
        }
        Predicate::All(inner) => {
            let mut b = Bitset::full(n);
            for p in inner {
                b.and(&eval(p, table, n)?);
            }
            b
        }
        Predicate::Present { column: name } => {
            let c = column(table, name)?;
            Bitset::from_fn(n, |i| c.present(i))
        }
        Predicate::Eq {
            column: name,
            value,
        } => {
            let c = column(table, name)?;
            match &c.data {
                ColumnData::Enum { dict, codes } => match code_of(dict, value) {
                    Some(code) => Bitset::from_fn(n, |i| c.present(i) && codes[i] == code),
                    None => Bitset::empty(n),
                },
                ColumnData::Text(v) => {
                    let want = value.to_lowercase();
                    Bitset::from_fn(n, |i| c.present(i) && v[i].to_lowercase() == want)
                }
                ColumnData::Bool(v) => {
                    let want = match value.to_lowercase().as_str() {
                        "true" | "yes" | "1" => true,
                        "false" | "no" | "0" => false,
                        _ => return Err(err(format!("{value:?} is not a boolean for {name}"))),
                    };
                    Bitset::from_fn(n, |i| c.present(i) && v[i] == want)
                }
                ColumnData::Int(_) | ColumnData::Real(_) => {
                    let x: f64 = value
                        .parse()
                        .map_err(|_| err(format!("{value:?} is not a number for {name}")))?;
                    range(c, n, Some((x, true)), Some((x, true)))?
                }
                ColumnData::Uuid(v) => {
                    let want = value.replace('-', "").to_lowercase();
                    Bitset::from_fn(n, |i| c.present(i) && hex(&v[i]) == want)
                }
                ColumnData::EnumList { .. } => {
                    return Err(err(format!("{name} is a list: use `has`")));
                }
            }
        }
        Predicate::Contains {
            column: name,
            value,
        } => {
            let c = column(table, name)?;
            let ColumnData::Text(v) = &c.data else {
                return Err(err(format!("{name} is not a text column")));
            };
            let want = value.to_lowercase();
            Bitset::from_fn(n, |i| c.present(i) && v[i].to_lowercase().contains(&want))
        }
        Predicate::Has {
            column: name,
            value,
        } => {
            let c = column(table, name)?;
            let ColumnData::EnumList { dict, .. } = &c.data else {
                return Err(err(format!("{name} is not a list column")));
            };
            let held = |code: u32| Bitset::from_fn(n, |i| c.data.codes_at(i).contains(&code));
            match code_of(dict, value) {
                Some(code) => held(code),
                // `rg` on a dictionary of single letters (colors) holds
                // every letter; any letter the dictionary lacks keeps none,
                // and so does no letter at all.
                None if !value.is_empty() && dict.values.iter().all(|v| v.chars().count() == 1) => {
                    let mut b = Bitset::full(n);
                    for ch in value.chars() {
                        match code_of(dict, ch.encode_utf8(&mut [0; 4])) {
                            Some(code) => b.and(&held(code)),
                            None => return Ok(Bitset::empty(n)),
                        }
                    }
                    b
                }
                None => Bitset::empty(n),
            }
        }
        Predicate::Range {
            column: name,
            min,
            max,
        } => range(column(table, name)?, n, *min, *max)?,
    })
}

fn range(
    c: &Column,
    n: usize,
    min: Option<(f64, bool)>,
    max: Option<(f64, bool)>,
) -> Result<Bitset, RuntimeError> {
    let within = |x: f64| {
        min.is_none_or(|(m, incl)| if incl { x >= m } else { x > m })
            && max.is_none_or(|(m, incl)| if incl { x <= m } else { x < m })
    };
    Ok(match &c.data {
        ColumnData::Int(v) => Bitset::from_fn(n, |i| c.present(i) && within(v[i] as f64)),
        ColumnData::Real(v) => Bitset::from_fn(n, |i| c.present(i) && within(v[i])),
        ColumnData::Enum { dict, codes } => {
            let Some(nums) = &dict.numeric else {
                return Err(err(format!("{} has no numeric values for a range", c.name)));
            };
            Bitset::from_fn(n, |i| {
                c.present(i)
                    && nums
                        .get(codes[i] as usize)
                        .copied()
                        .flatten()
                        .is_some_and(within)
            })
        }
        _ => return Err(err(format!("{} is not numeric", c.name))),
    })
}

/// the dictionary code of `value`: the exact entry, else the one that
/// matches ignoring ascii case (`r` finds `R`, `Common` finds `common`).
fn code_of(dict: &urna_format::sections::Dictionary, value: &str) -> Option<u32> {
    dict.code_of(value).or_else(|| {
        dict.values
            .iter()
            .position(|v| v.eq_ignore_ascii_case(value))
            .map(|i| i as u32)
    })
}

fn hex(u: &[u8; 16]) -> String {
    u.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use urna_format::sections::Dictionary;

    fn table() -> ScalarTable {
        ScalarTable {
            n_rows: 4,
            columns: vec![
                Column {
                    name: "power".into(),
                    validity: Some(vec![true, true, true, false]),
                    data: ColumnData::Enum {
                        dict: Dictionary {
                            values: vec!["*".into(), "1".into(), "3".into()],
                            numeric: Some(vec![None, Some(1.0), Some(3.0)]),
                        },
                        codes: vec![2, 0, 1, 0],
                    },
                },
                Column {
                    name: "type_line".into(),
                    validity: None,
                    data: ColumnData::Text(vec![
                        "Creature — Bear".into(),
                        "Instant".into(),
                        "Creature — Elf".into(),
                        "Land".into(),
                    ]),
                },
                Column {
                    name: "colors".into(),
                    validity: None,
                    data: ColumnData::EnumList {
                        dict: Dictionary {
                            values: vec!["G".into(), "R".into()],
                            numeric: None,
                        },
                        offsets: vec![0, 1, 2, 4, 4],
                        codes: vec![0, 1, 0, 1],
                    },
                },
                Column {
                    name: "cmc".into(),
                    validity: None,
                    data: ColumnData::Real(vec![2.0, 1.0, 1.0, 0.0]),
                },
            ],
        }
    }

    fn ones(f: Filter) -> Vec<usize> {
        f.bitset(&table(), 4).expect("filter").ones().collect()
    }

    #[test]
    fn predicates_narrow_and_compose() {
        let creature = Predicate::Contains {
            column: "type_line".into(),
            value: "creature".into(),
        };
        assert_eq!(ones(Filter::default().and(creature.clone())), vec![0, 2]);
        let pow3 = Predicate::Range {
            column: "power".into(),
            min: Some((3.0, true)),
            max: None,
        };
        assert_eq!(ones(Filter::default().and(pow3.clone())), vec![0]);
        let green = Predicate::Has {
            column: "colors".into(),
            value: "G".into(),
        };
        assert_eq!(
            ones(Filter::default().and(creature).and(green.clone())),
            vec![0, 2]
        );
        assert_eq!(
            ones(Filter::default().and(Predicate::Not(Box::new(green)))),
            vec![1, 3]
        );
        let has = |v: &str| Predicate::Has {
            column: "colors".into(),
            value: v.into(),
        };
        assert_eq!(ones(Filter::default().and(has("rg"))), vec![2]);
        assert_eq!(ones(Filter::default().and(has("g"))), vec![0, 2]);
        assert_eq!(ones(Filter::default().and(has("gx"))), Vec::<usize>::new());
        assert_eq!(ones(Filter::default().and(has(""))), Vec::<usize>::new());
        let cmc1 = Predicate::Eq {
            column: "cmc".into(),
            value: "1".into(),
        };
        assert_eq!(ones(Filter::default().and(cmc1)), vec![1, 2]);
        let star = Predicate::Eq {
            column: "power".into(),
            value: "*".into(),
        };
        assert_eq!(ones(Filter::default().and(star)), vec![1]);
        assert_eq!(ones(Filter::default()), vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_missing_column_or_a_wrong_kind_is_an_error() {
        let missing = Filter::default().and(Predicate::Eq {
            column: "nope".into(),
            value: "x".into(),
        });
        assert!(missing.bitset(&table(), 4).is_err());
        let wrong = Filter::default().and(Predicate::Range {
            column: "type_line".into(),
            min: Some((1.0, true)),
            max: None,
        });
        assert!(wrong.bitset(&table(), 4).is_err());
    }
}
