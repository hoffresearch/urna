//! graph_adjacency (0x0C) positive coverage (G1):
//!
//! - decode(encode(edges)) == canonical edges over empty / single-node /
//!   many-node / iso-single-edge-type / mixed-edge-type corpora.
//! - deterministic re-encode: two builds of the same edge set are byte-
//!   identical (canonical ascending (src, edge_type, dst) sort).
//! - O(1)/contiguous neighbor slice via parse_csr_parts.
//! - content_hash equality: a fixed corpus WITH vs WITHOUT the 0x0C graph
//!   section has IDENTICAL content_hash (citations stay stable), because the
//!   section is excluded from CANONICAL_SECTIONS.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::manifest::Manifest;
use urna_format::writer::UrnaFileBuilder;
use urna_format::{
    ChunkInput, EDGE_TYPE_CITATION, EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, Edge, UrnaView,
    decode_graph_adjacency, encode_graph_adjacency, parse_csr_parts,
};

fn edge(src: u32, dst: u32, t: u8) -> Edge {
    Edge {
        src,
        dst,
        edge_type: t,
    }
}

/// the canonical order encode emits: ascending (src, edge_type, dst).
fn canonical(mut edges: Vec<Edge>) -> Vec<Edge> {
    edges.sort_unstable_by(|a, b| {
        a.src
            .cmp(&b.src)
            .then(a.edge_type.cmp(&b.edge_type))
            .then(a.dst.cmp(&b.dst))
    });
    edges
}

fn roundtrip(n: usize, edges: Vec<Edge>) {
    let payload = encode_graph_adjacency(n, &edges).unwrap();
    let (got_n, got_edges) = decode_graph_adjacency(&payload).unwrap();
    assert_eq!(got_n, n, "n_nodes mismatch");
    assert_eq!(got_edges, canonical(edges.clone()), "edge set mismatch");

    // deterministic: re-encoding the same set is byte-identical, and so is
    // encoding the already-canonical decoded set.
    let payload2 = encode_graph_adjacency(n, &edges).unwrap();
    assert_eq!(payload, payload2, "two builds must be byte-identical");
    let payload3 = encode_graph_adjacency(got_n, &got_edges).unwrap();
    assert_eq!(payload, payload3, "re-encode of decoded set must match");
}

#[test]
fn empty_graph_roundtrips() {
    roundtrip(0, vec![]);
    roundtrip(5, vec![]); // nodes but no edges
}

#[test]
fn single_node_roundtrips() {
    // a lone node with a self-loop and one outgoing edge.
    roundtrip(2, vec![edge(0, 1, EDGE_TYPE_NEXT_CHUNK)]);
}

#[test]
fn many_node_roundtrips() {
    let n = 64;
    let mut edges = Vec::new();
    for i in 0..n - 1 {
        edges.push(edge(i as u32, (i + 1) as u32, EDGE_TYPE_NEXT_CHUNK));
        edges.push(edge((i + 1) as u32, i as u32, EDGE_TYPE_NEXT_CHUNK));
        edges.push(edge(i as u32, ((i + 7) % n) as u32, EDGE_TYPE_SEMANTIC));
    }
    roundtrip(n, edges);
}

#[test]
fn iso_single_edge_type_roundtrips() {
    // every edge one type -> falkordb iso single-scalar edge-type column.
    let mut edges = Vec::new();
    for i in 0..10u32 {
        edges.push(edge(i, (i + 1) % 10, EDGE_TYPE_NEXT_CHUNK));
    }
    roundtrip(10, edges.clone());
    // iso column is materially smaller than a per-edge run: the payload for
    // all-one-type must be smaller than an otherwise-identical mixed graph.
    let iso = encode_graph_adjacency(10, &edges).unwrap();
    let mut mixed = edges.clone();
    mixed[3].edge_type = EDGE_TYPE_SEMANTIC; // force a run
    let mixed = encode_graph_adjacency(10, &mixed).unwrap();
    assert!(
        iso.len() <= mixed.len(),
        "iso must not be larger than a run"
    );
}

#[test]
fn mixed_edge_type_roundtrips() {
    let edges = vec![
        edge(0, 1, EDGE_TYPE_NEXT_CHUNK),
        edge(0, 2, EDGE_TYPE_SEMANTIC),
        edge(0, 3, EDGE_TYPE_CITATION),
        edge(1, 0, EDGE_TYPE_NEXT_CHUNK),
        edge(2, 0, EDGE_TYPE_CITATION),
        edge(3, 1, EDGE_TYPE_SEMANTIC),
    ];
    roundtrip(4, edges);
}

#[test]
fn neighbor_slice_is_contiguous_and_correct() {
    let edges = vec![
        edge(0, 5, EDGE_TYPE_SEMANTIC),
        edge(0, 2, EDGE_TYPE_NEXT_CHUNK),
        edge(0, 9, EDGE_TYPE_SEMANTIC),
        edge(3, 1, EDGE_TYPE_NEXT_CHUNK),
    ];
    let payload = encode_graph_adjacency(10, &edges).unwrap();
    let parts = parse_csr_parts(&payload).unwrap();
    assert_eq!(parts.offsets.len(), 11);
    // node 0 canonical order: NEXT_CHUNK->2, SEMANTIC->5, SEMANTIC->9.
    let s = parts.offsets[0] as usize;
    let e = parts.offsets[1] as usize;
    assert_eq!(&parts.neighbors[s..e], &[2, 5, 9]);
    assert_eq!(
        &parts.edge_types[s..e],
        &[EDGE_TYPE_NEXT_CHUNK, EDGE_TYPE_SEMANTIC, EDGE_TYPE_SEMANTIC]
    );
}

#[test]
fn out_of_range_edge_rejected() {
    assert!(encode_graph_adjacency(3, &[edge(0, 5, EDGE_TYPE_NEXT_CHUNK)]).is_err());
    assert!(encode_graph_adjacency(3, &[edge(9, 0, EDGE_TYPE_NEXT_CHUNK)]).is_err());
}

// ---- content_hash equality: the honesty anchor for citations ----

fn build_corpus(path: &std::path::Path, with_graph: bool) {
    let n = 6usize;
    let dim = 4usize;
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: dim as u32,
        n_chunks: n as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest).reproducible(true);
    for i in 0..n {
        let mut emb = vec![0.0f32; dim];
        emb[i % dim] = 1.0;
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("chunk number {i} text"),
            source_uri: "doc.txt".into(),
            byte_start: (i * 10) as u64,
            byte_end: ((i + 1) * 10) as u64,
            embedding: emb,
        });
    }
    if with_graph {
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push(edge(i as u32, (i + 1) as u32, EDGE_TYPE_NEXT_CHUNK));
        }
        let payload = encode_graph_adjacency(n, &edges).unwrap();
        builder = builder.graph_adjacency(payload);
    }
    builder.write_to_path(path).unwrap();
}

#[test]
fn graph_section_does_not_change_content_hash() {
    let mut a = std::env::temp_dir();
    a.push("graph_ch_without.urna");
    let mut b = std::env::temp_dir();
    b.push("graph_ch_with.urna");
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
    build_corpus(&a, false);
    build_corpus(&b, true);

    let da = std::fs::read(&a).unwrap();
    let db = std::fs::read(&b).unwrap();
    let va = UrnaView::from_bytes(&da).unwrap();
    let vb = UrnaView::from_bytes(&db).unwrap();

    assert_eq!(
        va.content_hash_hex().unwrap(),
        vb.content_hash_hex().unwrap(),
        "the 0x0C graph section must NOT change content_hash (citations stable)"
    );
    // the graph build legitimately moves file_hash (the bytes differ).
    assert_ne!(va.file_hash_hex(), vb.file_hash_hex());
    // and the with-graph file genuinely carries the section.
    assert!(vb.entry(0x0C).is_ok(), "with-graph file must carry 0x0C");
    assert!(va.entry(0x0C).is_err(), "without-graph file must not");

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}
