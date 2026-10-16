//! chunk_scalars from JSON rows and a field spec: the one path the `urna
//! scalars` verb and the forge's `[scalars]` table share. One spec per
//! column; the kind decides how a value is read; dictionaries are sorted,
//! so the same rows give the same bytes. The kind `enum_map` turns an
//! object into one enum column per key, `name.key`; the kind `chunk`
//! writes no column, only a schema entry that points at the stored text.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use super::{Column, ColumnData, Dictionary, ScalarTable};
use crate::error::UrnaError;
use crate::manifest::schema::SchemaField;

/// One column to write, as a schema file or a spec declares it.
#[derive(Clone, Debug, Deserialize)]
pub struct FieldSpec {
    pub name: String,
    pub kind: String,
    /// the row key to read; the column name when absent.
    #[serde(default)]
    pub source: Option<String>,
    /// an enum whose values are numbers (`"3"`) gets a numeric table, so
    /// a range predicate works; values without a number (`"*"`) get none.
    #[serde(default)]
    pub numeric: bool,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub display: Option<String>,
}

impl FieldSpec {
    fn source(&self) -> &str {
        self.source.as_deref().unwrap_or(&self.name)
    }

    fn field(&self, name: &str, kind: &str) -> SchemaField {
        SchemaField {
            name: name.to_string(),
            kind: kind.to_string(),
            label: self.label.clone(),
            unit: self.unit.clone(),
            roles: self.roles.clone(),
            prefix: self.prefix.clone(),
            display: self.display.clone(),
        }
    }

    /// the schema entry of one `name.key` column of an enum_map: filtered
    /// through its group's prefix and drawn by its group, so it keeps only
    /// the roles that are per column.
    fn sub_field(&self, name: &str) -> SchemaField {
        let mut f = self.field(name, "enum");
        f.roles.retain(|r| r == "filter");
        f.display = None;
        f.label = None;
        f
    }
}

fn invalid(msg: String) -> UrnaError {
    UrnaError::InvalidInput(msg)
}

/// The rows of a file: a bare array, or an object holding one under
/// `rows` or `cards`, with an optional `content_hash` binding it to a
/// corpus. A row may carry `ordinal`, which must be its position.
pub fn records(doc: Value) -> Result<(Option<String>, Vec<Value>), UrnaError> {
    let (bound, rows) = match doc {
        Value::Array(rows) => (None, rows),
        Value::Object(mut o) => {
            let bound = o
                .get("content_hash")
                .and_then(Value::as_str)
                .map(str::to_string);
            let rows = o
                .remove("rows")
                .or_else(|| o.remove("cards"))
                .and_then(|v| v.as_array().cloned());
            match rows {
                Some(rows) => (bound, rows),
                None => return Err(invalid("the rows hold no `rows` or `cards` array".into())),
            }
        }
        _ => {
            return Err(invalid(
                "the rows are neither an array nor an object".into(),
            ));
        }
    };
    for (i, r) in rows.iter().enumerate() {
        if let Some(o) = r.get("ordinal").and_then(Value::as_u64)
            && o as usize != i
        {
            return Err(invalid(format!("row {i} carries ordinal {o}")));
        }
    }
    Ok((bound, rows))
}

/// The schema fields for columns a file already carries: every spec must
/// name a column of `table` of its kind (or, for `enum_map`, at least one
/// `name.key`).
pub fn declare(specs: &[FieldSpec], table: &ScalarTable) -> Result<Vec<SchemaField>, UrnaError> {
    let mut fields = Vec::new();
    for spec in specs {
        match spec.kind.as_str() {
            "enum_map" => {
                let prefix = format!("{}.", spec.name);
                let keys: Vec<&str> = table
                    .columns
                    .iter()
                    .filter_map(|c| c.name.strip_prefix(&prefix))
                    .collect();
                if keys.is_empty() {
                    return Err(invalid(format!("no `{}.*` columns in the file", spec.name)));
                }
                fields.push(spec.field(&spec.name, "enum_map"));
                for key in keys {
                    fields.push(spec.sub_field(&format!("{prefix}{key}")));
                }
            }
            "chunk" => fields.push(spec.field(&spec.name, "chunk")),
            kind => {
                let Some(c) = table.column(&spec.name) else {
                    return Err(invalid(format!("no column {} in the file", spec.name)));
                };
                if c.data.kind().as_str() != kind {
                    return Err(invalid(format!(
                        "column {} is {} in the file, the schema says {kind}",
                        spec.name,
                        c.data.kind().as_str()
                    )));
                }
                fields.push(spec.field(&spec.name, kind));
            }
        }
    }
    Ok(fields)
}

/// Build the table and the schema fields from `specs` over `rows`.
pub fn build(
    specs: &[FieldSpec],
    rows: &[Value],
) -> Result<(ScalarTable, Vec<SchemaField>), UrnaError> {
    let mut columns = Vec::new();
    let mut fields = Vec::new();
    for spec in specs {
        let values: Vec<&Value> = rows
            .iter()
            .map(|r| r.get(spec.source()).unwrap_or(&Value::Null))
            .collect();
        match spec.kind.as_str() {
            "enum_map" => {
                let mut keys: Vec<String> = Vec::new();
                for v in &values {
                    if let Some(o) = v.as_object() {
                        for k in o.keys() {
                            if !keys.contains(k) {
                                keys.push(k.clone());
                            }
                        }
                    }
                }
                keys.sort();
                fields.push(spec.field(&spec.name, "enum_map"));
                for key in keys {
                    let name = format!("{}.{key}", spec.name);
                    let sub: Vec<&Value> = values
                        .iter()
                        .map(|v| v.get(&key).unwrap_or(&Value::Null))
                        .collect();
                    columns.push(enum_column(&name, &sub, false)?);
                    fields.push(spec.sub_field(&name));
                }
            }
            "chunk" => fields.push(spec.field(&spec.name, "chunk")),
            kind => {
                columns.push(column(&spec.name, kind, &values, spec.numeric)?);
                fields.push(spec.field(&spec.name, kind));
            }
        }
    }
    Ok((
        ScalarTable {
            n_rows: rows.len(),
            columns,
        },
        fields,
    ))
}

/// one value per row: `absent` with a false flag for a null, the parsed
/// value for anything `parse` accepts, an error naming the column else.
fn scalar<T: Clone>(
    name: &str,
    kind: &str,
    values: &[&Value],
    absent: T,
    parse: impl Fn(&Value) -> Option<T>,
) -> Result<(Vec<T>, Option<Vec<bool>>), UrnaError> {
    let mut out = Vec::with_capacity(values.len());
    let mut ok = Vec::with_capacity(values.len());
    for v in values {
        if v.is_null() {
            out.push(absent.clone());
            ok.push(false);
        } else if let Some(x) = parse(v) {
            out.push(x);
            ok.push(true);
        } else {
            return Err(invalid(format!("column {name}: {v} is not {kind}")));
        }
    }
    let validity = if ok.iter().all(|&b| b) {
        None
    } else {
        Some(ok)
    };
    Ok((out, validity))
}

/// a column of `kind` read from `values`.
fn column(name: &str, kind: &str, values: &[&Value], numeric: bool) -> Result<Column, UrnaError> {
    let (data, validity) = match kind {
        "int" => {
            let (v, ok) = scalar(name, kind, values, 0i64, Value::as_i64)?;
            (ColumnData::Int(v), ok)
        }
        "real" => {
            let (v, ok) = scalar(name, kind, values, 0.0f64, Value::as_f64)?;
            (ColumnData::Real(v), ok)
        }
        "bool" => {
            let (v, ok) = scalar(name, kind, values, false, Value::as_bool)?;
            (ColumnData::Bool(v), ok)
        }
        "text" => {
            let (v, ok) = scalar(name, kind, values, String::new(), |v| {
                v.as_str().map(str::to_string)
            })?;
            (ColumnData::Text(v), ok)
        }
        "uuid" => {
            let (v, ok) = scalar(name, kind, values, [0u8; 16], |v| {
                v.as_str().and_then(parse_uuid)
            })?;
            (ColumnData::Uuid(v), ok)
        }
        "enum" => return enum_column(name, values, numeric),
        "enum_list" => {
            let mut lists: Vec<Vec<String>> = Vec::with_capacity(values.len());
            for v in values {
                match v {
                    Value::Array(items) => {
                        let mut l = Vec::with_capacity(items.len());
                        for it in items {
                            let Some(s) = it.as_str() else {
                                return Err(invalid(format!(
                                    "column {name}: {it} in a list is not a string"
                                )));
                            };
                            l.push(s.to_string());
                        }
                        lists.push(l);
                    }
                    Value::Null => lists.push(Vec::new()),
                    other => return Err(invalid(format!("column {name}: {other} is not a list"))),
                }
            }
            let dict = dictionary(lists.iter().flatten().map(String::as_str), false);
            let mut offsets = Vec::with_capacity(values.len() + 1);
            let mut codes = Vec::new();
            offsets.push(0u32);
            for l in &lists {
                for s in l {
                    codes.push(code_of(&dict, s));
                }
                offsets.push(codes.len() as u32);
            }
            (
                ColumnData::EnumList {
                    dict,
                    offsets,
                    codes,
                },
                None,
            )
        }
        other => return Err(invalid(format!("column {name}: unknown kind {other:?}"))),
    };
    Ok(Column {
        name: name.into(),
        validity,
        data,
    })
}

fn enum_column(name: &str, values: &[&Value], numeric: bool) -> Result<Column, UrnaError> {
    let mut strings: Vec<Option<String>> = Vec::with_capacity(values.len());
    for v in values {
        strings.push(match v {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            Value::Number(x) => Some(x.to_string()),
            other => {
                return Err(invalid(format!(
                    "column {name}: {other} is not an enum value"
                )));
            }
        });
    }
    let dict = dictionary(strings.iter().flatten().map(String::as_str), numeric);
    let codes: Vec<u32> = strings
        .iter()
        .map(|s| s.as_deref().map_or(0, |s| code_of(&dict, s)))
        .collect();
    let ok: Vec<bool> = strings.iter().map(Option::is_some).collect();
    Ok(Column {
        name: name.into(),
        validity: if ok.iter().all(|&b| b) {
            None
        } else {
            Some(ok)
        },
        data: ColumnData::Enum { dict, codes },
    })
}

fn dictionary<'a>(values: impl Iterator<Item = &'a str>, numeric: bool) -> Dictionary {
    let set: BTreeMap<&str, ()> = values.map(|v| (v, ())).collect();
    let values: Vec<String> = set.keys().map(|s| s.to_string()).collect();
    let numeric = numeric.then(|| values.iter().map(|v| v.parse::<f64>().ok()).collect());
    Dictionary { values, numeric }
}

fn code_of(dict: &Dictionary, value: &str) -> u32 {
    // the dictionary was built from these very values, so every lookup
    // lands; a miss would be a bug, not an input error.
    dict.values
        .binary_search_by(|v| v.as_str().cmp(value))
        .map_or(0, |i| i as u32)
}

fn parse_uuid(s: &str) -> Option<[u8; 16]> {
    let hex: String = s.chars().filter(|c| *c != '-').collect();
    if hex.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_become_typed_columns_with_sorted_dictionaries() {
        let specs: Vec<FieldSpec> = serde_json::from_str(
            r#"[{"name":"power","kind":"enum","numeric":true,"prefix":"pow"},
                {"name":"legal","kind":"enum_map","prefix":"f","roles":["filter","display"]},
                {"name":"colors","kind":"enum_list"},
                {"name":"id","kind":"uuid","source":"oracle_id"},
                {"name":"cmc","kind":"real"},
                {"name":"rules","kind":"chunk","unit":"3"}]"#,
        )
        .expect("specs");
        let rows: Vec<Value> = serde_json::from_str(
            r#"[{"power":"3","legal":{"pauper":"legal"},"colors":["R","G"],"oracle_id":"8ae3562f-28b7-4462-96ed-be0cf7052ccc","cmc":2},
                {"power":null,"legal":{"modern":"banned","pauper":"legal"},"colors":[],"oracle_id":null,"cmc":null}]"#,
        )
        .expect("rows");
        let (table, fields) = build(&specs, &rows).expect("builds");
        assert_eq!(table.n_rows, 2);
        let names: Vec<&str> = table.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "power",
                "legal.modern",
                "legal.pauper",
                "colors",
                "id",
                "cmc"
            ]
        );
        let power = &table.columns[0];
        assert!(power.present(0) && !power.present(1));
        assert_eq!(
            power.data.dictionary().and_then(|d| d.numeric_of(0)),
            Some(3.0)
        );
        let colors = table.column("colors").expect("colors");
        assert_eq!(
            colors.data.dictionary().map(|d| d.values.clone()),
            Some(vec!["G".into(), "R".into()])
        );
        assert_eq!(colors.data.codes_at(0), &[1, 0]);
        assert!(
            fields
                .iter()
                .any(|f| f.name == "legal" && f.kind == "enum_map")
        );
        assert!(fields.iter().any(|f| f.name == "legal.pauper"
            && f.prefix.as_deref() == Some("f")
            && f.roles == ["filter"]));
        assert!(
            fields
                .iter()
                .any(|f| f.name == "rules" && f.kind == "chunk")
        );
        assert_eq!(declare(&specs, &table).expect("declares"), fields);
        let (bound, recs) =
            records(serde_json::json!({"content_hash":"sha256:x","cards":[{"ordinal":0}]}))
                .expect("records");
        assert_eq!((bound.as_deref(), recs.len()), (Some("sha256:x"), 1));
        assert!(records(serde_json::json!({"cards":[{"ordinal":1}]})).is_err());
        let bad: Vec<Value> = serde_json::from_str(r#"[{"cmc":"two"}]"#).expect("rows");
        assert!(build(&specs[4..5], &bad).is_err());
    }
}
