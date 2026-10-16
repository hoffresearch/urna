//! a filter resolved from chunk_scalars is applied before scoring on every
//! path: the exact kept scan equals the brute force over the kept rows, the
//! hnsw path under a filter never comes back short while kept chunks remain
//! and agrees with the exact kept scan when the kept set is small, the
//! hybrid path keeps only kept chunks, and the explain line says how many
//! were kept. the search syntax resolves over the manifest schema.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use std::path::PathBuf;
use urna_engine::MmapUrnaFile;
use urna_engine::ann::HnswIndex;
use urna_engine::scalars::{Filter, Predicate, parse};
use urna_format::ChunkInput;
use urna_format::manifest::Manifest;
use urna_format::manifest::schema::SchemaField;
use urna_format::sections::{Column, ColumnData, Dictionary, ScalarTable, encode_chunk_scalars};
use urna_format::writer::UrnaFileBuilder;

const N: usize = 600;
const DIM: usize = 16;

fn vectors() -> Vec<Vec<f32>> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    (0..N)
        .map(|_| {
            let mut v: Vec<f32> = (0..DIM)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed % 2000) as f32 / 1000.0) - 1.0
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn build(path: &PathBuf) -> Vec<Vec<f32>> {
    let vecs = vectors();
    let manifest = Manifest {
        embedding_model: "demo".into(),
        embedding_dim: DIM as u32,
        n_chunks: N as u64,
        chunker_version: "demo-chunker/1".into(),
        model_hash: format!("sha256:{}", "0".repeat(64)),
        schema: Some(vec![
            SchemaField {
                name: "year".into(),
                kind: "int".into(),
                prefix: Some("y".into()),
                roles: vec!["filter".into()],
                ..Default::default()
            },
            SchemaField {
                name: "tag".into(),
                kind: "enum".into(),
                prefix: Some("t".into()),
                roles: vec!["filter".into()],
                ..Default::default()
            },
        ]),
        ..Default::default()
    };
    let mut builder = UrnaFileBuilder::new(manifest).reproducible(true);
    for (i, v) in vecs.iter().enumerate() {
        builder = builder.add_chunk(ChunkInput {
            canonical_text: format!("chunk {i} {}", if i % 3 == 0 { "law" } else { "art" }),
            source_uri: "docs".into(),
            byte_start: i as u64,
            byte_end: i as u64 + 1,
            embedding: v.clone(),
        });
    }
    let flat: Vec<u8> = vecs
        .iter()
        .flatten()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let hnsw = HnswIndex::build_from_f32(&flat, N, DIM, 16, 64, 7);
    builder = builder.hnsw_index(hnsw.to_bytes());
    let table = ScalarTable {
        n_rows: N,
        columns: vec![
            Column {
                name: "year".into(),
                validity: None,
                data: ColumnData::Int((0..N as i64).map(|i| 1990 + i % 40).collect()),
            },
            Column {
                name: "tag".into(),
                validity: None,
                data: ColumnData::Enum {
                    dict: Dictionary {
                        values: vec!["art".into(), "law".into()],
                        numeric: None,
                    },
                    codes: (0..N as u32).map(|i| u32::from(i % 3 == 0)).collect(),
                },
            },
        ],
    };
    builder = builder.chunk_scalars(encode_chunk_scalars(&table).unwrap());
    builder.write_to_path(path).unwrap();
    vecs
}

fn brute(vecs: &[Vec<f32>], q: &[f32], keep: impl Fn(usize) -> bool, k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vecs
        .iter()
        .enumerate()
        .filter(|(i, _)| keep(*i))
        .map(|(i, v)| (i, v.iter().zip(q).map(|(a, b)| a * b).sum()))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    scored.into_iter().take(k).map(|p| p.0).collect()
}

#[test]
fn filtered_paths_agree_with_brute_force_over_the_kept_rows() {
    let path = std::env::temp_dir().join("search_filtered.urna");
    let vecs = build(&path);
    let rt = MmapUrnaFile::open(&path).unwrap();
    let scalars = rt.scalars().expect("columns opened behind scalars_present");
    assert_eq!(scalars.table.columns.len(), 2);
    assert_eq!(scalars.schema.len(), 2);

    let filter = Filter::default()
        .and(Predicate::Range {
            column: "year".into(),
            min: Some((2010.0, true)),
            max: None,
        })
        .and(Predicate::Eq {
            column: "tag".into(),
            value: "law".into(),
        });
    let keep = rt.keep(&filter).unwrap();
    let kept = keep.count();
    let want = |i: usize| 1990 + (i as i64) % 40 >= 2010 && i % 3 == 0;
    assert_eq!(kept, (0..N).filter(|&i| want(i)).count());

    let q = &vecs[7];
    let exact = rt.search_where(q, 10, &keep).unwrap();
    let ids = |r: &urna_engine::SearchResult| -> Vec<usize> {
        r.hits
            .iter()
            .map(|h| {
                rt.chunk_ids()
                    .iter()
                    .position(|c| *c == h.chunk_id)
                    .unwrap()
            })
            .collect()
    };
    assert_eq!(ids(&exact), brute(&vecs, q, want, 10));
    assert_eq!(exact.explain.filter_kept, Some(kept));
    assert!(exact.hits.iter().all(|h| {
        let i = rt
            .chunk_ids()
            .iter()
            .position(|c| *c == h.chunk_id)
            .unwrap();
        want(i)
    }));

    // the kept set is small, so the ann path takes the exact kept scan.
    let ann = rt.search_ann_where(q, 10, 64, &keep).unwrap();
    assert_eq!(ids(&ann), ids(&exact));
    assert_eq!(ann.hits.len(), 10);

    // the hybrid path under the filter (no bm25 section: vector path only)
    // keeps only kept chunks and is never short.
    let hybrid = rt.search_hybrid_where(q, "law", 10, 32, &keep).unwrap();
    assert_eq!(hybrid.hits.len(), 10);
    assert_eq!(hybrid.explain.filter_kept, Some(kept));
    assert!(ids(&hybrid).iter().all(|&i| want(i)));

    // routed: without a filter it is the plain route; with one, the kept one.
    let routed = rt
        .search_routed_where(q, None, 10, 64, Some(&keep))
        .unwrap();
    assert_eq!(ids(&routed), ids(&exact));
    let plain = rt.search_routed_where(q, None, 10, 64, None).unwrap();
    assert_eq!(plain.explain.filter_kept, None);

    // a filter that keeps nothing returns nothing, truncated false.
    let none = rt
        .keep(&Filter::default().and(Predicate::Eq {
            column: "tag".into(),
            value: "poem".into(),
        }))
        .unwrap();
    let empty = rt.search_where(q, 10, &none).unwrap();
    assert!(empty.hits.is_empty() && !empty.truncated);

    // the syntax resolves over the manifest schema.
    let parsed = parse("y>=2010 t:law chunk", &scalars.schema).unwrap();
    assert_eq!(parsed.text, "chunk");
    assert_eq!(rt.keep(&parsed.filter).unwrap(), keep);
    assert!(parse("z:1", &scalars.schema).is_err());
    assert!(
        rt.keep(&Filter::default().and(Predicate::Eq {
            column: "nope".into(),
            value: "x".into()
        }))
        .is_err()
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_corpus_without_columns_keeps_everything_for_an_empty_filter() {
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../format/tests/fixtures/golden_v1_minimal.urna");
    let rt = MmapUrnaFile::open(&golden).unwrap();
    assert!(rt.scalars().is_none());
    let all = rt.keep(&Filter::default()).unwrap();
    assert_eq!(all.count(), rt.n_embeddings());
    assert!(
        rt.keep(&Filter::default().and(Predicate::Present {
            column: "year".into()
        }))
        .is_err()
    );
}
