//! negative paths for the graph_adjacency (0x0C) csr codec (G1):
//! every malformed or hostile payload must return a typed `UrnaError`,
//! NEVER panic (mirrors negative_txt_streams discipline). plus an
//! exhaustive prefix-truncation fuzz.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::{
    EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, Edge, UrnaError, decode_graph_adjacency,
    encode_graph_adjacency, parse_csr_parts,
};

fn edge(src: u32, dst: u32, t: u8) -> Edge {
    Edge {
        src,
        dst,
        edge_type: t,
    }
}

fn good_packed() -> Vec<u8> {
    let n = 8;
    let mut edges = Vec::new();
    for i in 0..n - 1 {
        edges.push(edge(i as u32, (i + 1) as u32, EDGE_TYPE_NEXT_CHUNK));
        edges.push(edge((i + 1) as u32, i as u32, EDGE_TYPE_SEMANTIC));
    }
    encode_graph_adjacency(n, &edges).unwrap()
}

fn assert_err(res: Result<(usize, Vec<Edge>), UrnaError>) {
    assert!(res.is_err(), "expected a typed error, got Ok");
}

#[test]
fn baseline_decodes_cleanly() {
    let packed = good_packed();
    assert!(decode_graph_adjacency(&packed).is_ok());
    assert!(parse_csr_parts(&packed).is_ok());
}

#[test]
fn empty_payload_errors() {
    assert_err(decode_graph_adjacency(&[]));
    assert!(parse_csr_parts(&[]).is_err());
}

#[test]
fn bad_version_errors() {
    let mut packed = good_packed();
    // bump the leading u32 version to an unsupported value.
    packed[0] = packed[0].wrapping_add(7);
    match decode_graph_adjacency(&packed) {
        Err(UrnaError::UnsupportedSectionVersion { .. }) => {}
        other => panic!("expected UnsupportedSectionVersion, got {:?}", other),
    }
}

#[test]
fn oversized_n_nodes_errors() {
    // tamper the u64 n_nodes (bytes [4..12)) to a huge value: the offsets
    // length cross-check (offsets.len() != n_nodes+1) must reject, never
    // allocate gigabytes from the claim alone, never panic.
    let mut packed = good_packed();
    packed[4..12].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_err(decode_graph_adjacency(&packed));
}

#[test]
fn truncated_header_errors() {
    let packed = good_packed();
    for len in 0..12 {
        assert_err(decode_graph_adjacency(&packed[..len]));
    }
}

#[test]
fn truncated_columns_error_never_panic() {
    // chop into every byte region past the header (offsets, dst, edge-type):
    // every prefix must error or parse, never panic.
    let packed = good_packed();
    for cut in 12..packed.len() {
        let _ = decode_graph_adjacency(&packed[..cut]);
        let _ = parse_csr_parts(&packed[..cut]);
    }
}

#[test]
fn dropped_tail_byte_errors() {
    // remove the last edge-type byte: the column length / final-offset cross
    // checks (or the cursor EOF) must reject it.
    let mut packed = good_packed();
    packed.pop();
    assert_err(decode_graph_adjacency(&packed));
}

#[test]
fn degree_over_cap_via_tampered_offsets_does_not_panic() {
    // build a tiny graph then flip bytes throughout: the monotone-offset,
    // degree-cap, dst-range, and column-length checks must keep every result
    // a typed error or a (different but valid) parse, never a panic.
    let packed = good_packed();
    for i in 12..packed.len() {
        let mut evil = packed.clone();
        evil[i] ^= 0xFF;
        let _ = decode_graph_adjacency(&evil); // must not panic
        let _ = parse_csr_parts(&evil);
    }
}

#[test]
fn fuzz_every_truncation_never_panics() {
    // exhaustive prefix truncation: the core no-panic-on-hostile-mmap
    // guarantee (mirrors negative_txt_streams).
    let packed = good_packed();
    for cut in 0..=packed.len() {
        let _ = decode_graph_adjacency(&packed[..cut]);
        let _ = parse_csr_parts(&packed[..cut]);
    }
}
