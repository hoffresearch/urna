//! negative paths for the chunk_scalars (0x0D) codec: every malformed or
//! hostile payload must return a typed `UrnaError`, NEVER panic, and a
//! hostile claim must fail before it allocates. plus an exhaustive
//! prefix-truncation sweep over a payload of every column kind.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::UrnaError;
use urna_format::sections::{
    Column, ColumnData, Dictionary, ScalarTable, decode_chunk_scalars, encode_chunk_scalars,
};

const N: usize = 6;

fn table() -> ScalarTable {
    ScalarTable {
        n_rows: N,
        columns: vec![
            Column {
                name: "rank".into(),
                validity: Some(vec![true, false, true, true, false, true]),
                data: ColumnData::Int(vec![3, 0, -7, 1 << 33, 0, 12]),
            },
            Column {
                name: "cmc".into(),
                validity: None,
                data: ColumnData::Real(vec![0.0, 1.5, 2.0, 3.0, 4.0, 5.5]),
            },
            Column {
                name: "foil".into(),
                validity: None,
                data: ColumnData::Bool(vec![true, false, true, true, false, false]),
            },
            Column {
                name: "name".into(),
                validity: None,
                data: ColumnData::Text(
                    ["a", "", "Relâmpago", "d", "e", "f"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            },
            Column {
                name: "id".into(),
                validity: None,
                data: ColumnData::Uuid((0..N as u8).map(|i| [i; 16]).collect()),
            },
            Column {
                name: "power".into(),
                validity: Some(vec![true, true, true, true, true, false]),
                data: ColumnData::Enum {
                    dict: Dictionary {
                        values: vec!["*".into(), "1".into(), "3".into()],
                        numeric: Some(vec![None, Some(1.0), Some(3.0)]),
                    },
                    codes: vec![2, 0, 1, 1, 2, 0],
                },
            },
            Column {
                name: "colors".into(),
                validity: None,
                data: ColumnData::EnumList {
                    dict: Dictionary {
                        values: vec!["G".into(), "R".into(), "U".into()],
                        numeric: None,
                    },
                    offsets: vec![0, 2, 2, 3, 3, 5, 6],
                    codes: vec![0, 1, 2, 0, 1, 2],
                },
            },
        ],
    }
}

fn good() -> Vec<u8> {
    encode_chunk_scalars(&table()).unwrap()
}

fn is_typed(res: &Result<ScalarTable, UrnaError>) -> bool {
    matches!(
        res,
        Err(UrnaError::MalformedSectionPayload { .. } | UrnaError::UnsupportedSectionVersion { .. })
    )
}

#[test]
fn baseline_decodes_and_is_byte_stable() {
    let bytes = good();
    let back = decode_chunk_scalars(&bytes, N).unwrap();
    assert_eq!(back, table());
    assert_eq!(encode_chunk_scalars(&back).unwrap(), bytes);
}

#[test]
fn every_prefix_truncation_is_a_typed_error() {
    let bytes = good();
    for cut in 0..bytes.len() {
        let res = decode_chunk_scalars(&bytes[..cut], N);
        assert!(is_typed(&res), "prefix of {cut} bytes decoded or panicked");
    }
}

#[test]
fn a_row_count_other_than_the_files_is_refused() {
    let bytes = good();
    assert!(is_typed(&decode_chunk_scalars(&bytes, N - 1)));
    assert!(is_typed(&decode_chunk_scalars(&bytes, N + 1)));
    assert!(is_typed(&decode_chunk_scalars(&bytes, 0)));
}

#[test]
fn hostile_counts_fail_before_allocating() {
    let bytes = good();
    // the column count at [12..20): a claim far above the bytes behind it.
    let mut hostile = bytes.clone();
    hostile[12..20].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    assert!(is_typed(&decode_chunk_scalars(&hostile, N)));
    // the row count at [4..12): matches nothing the caller expects.
    let mut rows = bytes.clone();
    rows[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(is_typed(&decode_chunk_scalars(&rows, N)));
    // the first column's name length at [20..24).
    let mut name = bytes;
    name[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(is_typed(&decode_chunk_scalars(&name, N)));
}

#[test]
fn a_dictionary_code_out_of_range_is_refused() {
    // shrink the power dictionary to one entry: the codes 1 and 2 overflow.
    let bytes = good();
    let at = bytes
        .windows(5)
        .position(|w| w == b"power")
        .expect("power column");
    let dict_count = at + 5 + 1 + 1 + N.div_ceil(8) + 4;
    let mut bad = bytes;
    bad[dict_count..dict_count + 4].copy_from_slice(&1u32.to_le_bytes());
    assert!(is_typed(&decode_chunk_scalars(&bad, N)));
}

#[test]
fn offsets_out_of_order_and_bad_flags_are_refused() {
    let mut t = table();
    if let ColumnData::EnumList { offsets, .. } = &mut t.columns[6].data {
        offsets[2] = 9;
    }
    assert!(encode_chunk_scalars(&t).is_err());
    let bytes = good();
    // the first column's validity flag sits after its name and kind byte.
    let flag = 20 + 4 + 4 + 1;
    let mut bad = bytes.clone();
    bad[flag] = 7;
    assert!(is_typed(&decode_chunk_scalars(&bad, N)));
    let mut kind = bytes;
    kind[flag - 1] = 42;
    assert!(is_typed(&decode_chunk_scalars(&kind, N)));
}

#[test]
fn trailing_bytes_and_a_future_version_are_refused() {
    let mut extra = good();
    extra.push(0);
    assert!(is_typed(&decode_chunk_scalars(&extra, N)));
    let mut versioned = good();
    versioned[..4].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(
        decode_chunk_scalars(&versioned, N),
        Err(UrnaError::UnsupportedSectionVersion { version: 2, .. })
    ));
}

#[test]
fn random_byte_flips_never_panic() {
    let bytes = good();
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..2000 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let mut mutated = bytes.clone();
        let at = (seed % bytes.len() as u64) as usize;
        mutated[at] ^= (seed >> 8) as u8 | 1;
        let _ = decode_chunk_scalars(&mutated, N);
    }
}
