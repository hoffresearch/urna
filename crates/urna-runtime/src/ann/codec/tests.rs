//! Negative-decode tests for the HNSW payload.
//!
//! A crafted payload must yield a typed `Err`, never a panic or a giant
//! allocation. These live beside the codec (not under `crates/*/tests/`)
//! because they exercise the private decode helpers through
//! `HnswIndex::from_bytes`.

use super::*;

fn write_blob(out: &mut Vec<u8>, blob: &[u8]) {
    out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    out.extend_from_slice(blob);
}

fn v2_header(n_nodes: u32) -> Vec<u8> {
    let mut out = 2u32.to_le_bytes().to_vec(); // version
    for v in [16u32, 32, 400, 0, 0, n_nodes] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// the column-length checks alone do not bound the decoded level/degree
// VALUES, the per-value caps do.
#[test]
fn v2_rejects_hostile_level_value() {
    let mut p = v2_header(1);
    write_blob(&mut p, &pack_u64s(&[u64::MAX])); // level = u64::MAX
    write_blob(&mut p, &pack_u64s(&[]));
    write_blob(&mut p, &pack_u64s(&[]));
    assert!(HnswIndex::from_bytes(&p, 1, 8).is_err());
}

#[test]
fn v2_rejects_hostile_degree_value() {
    let mut p = v2_header(1);
    write_blob(&mut p, &pack_u64s(&[0])); // one node, one layer
    write_blob(&mut p, &pack_u64s(&[u64::MAX])); // degree = u64::MAX
    write_blob(&mut p, &pack_u64s(&[]));
    assert!(HnswIndex::from_bytes(&p, 1, 8).is_err());
}

#[test]
fn v2_truncated_columns_error_not_panic() {
    let mut p = v2_header(1);
    write_blob(&mut p, &pack_u64s(&[0]));
    // counts and ids columns missing -> EOF, typed error.
    assert!(HnswIndex::from_bytes(&p, 1, 8).is_err());
}

// a neighbour id parsed from a hostile file that is >= n_nodes would index
// `nodes` / the vector store out of range at search time. It must be
// rejected at decode, never accepted then panic on the first query.
#[test]
fn v2_rejects_out_of_range_neighbour_id() {
    let mut p = v2_header(2); // n_nodes = 2 -> valid ids are {0, 1}
    write_blob(&mut p, &pack_u64s(&[0, 0])); // two nodes, level 0
    write_blob(&mut p, &pack_u64s(&[1, 1])); // one neighbour each
    write_blob(&mut p, &pack_u64s(&[5, 0])); // id 5 is out of range
    assert!(HnswIndex::from_bytes(&p, 2, 8).is_err());
}

#[test]
fn v1_rejects_out_of_range_neighbour_id() {
    let mut out = 1u32.to_le_bytes().to_vec(); // version 1
    for v in [16u32, 32, 400, 0, 0, 1] {
        out.extend_from_slice(&v.to_le_bytes()); // entry 0, max_level 0, n 1
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // node 0 level
    out.extend_from_slice(&1u32.to_le_bytes()); // layer 0 degree = 1
    out.extend_from_slice(&7u32.to_le_bytes()); // neighbour id 7 >= n(1)
    assert!(HnswIndex::from_bytes(&out, 1, 8).is_err());
}

#[test]
fn rejects_out_of_range_entry_point() {
    let mut out = 2u32.to_le_bytes().to_vec(); // version 2
    for v in [16u32, 32, 400, 9, 0, 1] {
        out.extend_from_slice(&v.to_le_bytes()); // entry_point 9, n_nodes 1
    }
    write_blob(&mut out, &pack_u64s(&[0])); // one node, level 0
    write_blob(&mut out, &pack_u64s(&[0])); // that layer has 0 neighbours
    write_blob(&mut out, &pack_u64s(&[])); // no ids
    assert!(HnswIndex::from_bytes(&out, 1, 8).is_err());
}
