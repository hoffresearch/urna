//! Every zstd-compressed section of real `.urna` files, decoded by both
//! decoders: the fixtures git sees in the tree, and corpora built here with
//! the writer under a compressed preset (single-frame zstd, txt_streams,
//! and a dictionary-framed text section put in place with add_sections).
//! The dictionary-framed corpus is then opened by the engine and must
//! answer: the top hit of a chunk's own vector is that chunk, with its text.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test: a failure is the finding"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../clitui/tests/common/needs.rs"]
mod needs;

use urna_engine::MmapUrnaFile;
use urna_format::encoding::{encode_txt_streams, encode_zstd_dict, train_dict};
use urna_format::layout::{
    SECTION_CHUNKS_CANONICAL, SECTION_DICTIONARY, SECTION_ENCODING_RAW,
    SECTION_ENCODING_TXT_STREAMS, SECTION_ENCODING_ZSTD, SECTION_ENCODING_ZSTD_DICT,
};
use urna_format::manifest::{Capabilities, Manifest};
use urna_format::reader::UrnaView;
use urna_format::writer::{SectionEncoding, UrnaFileBuilder};
use urna_format::{ChunkInput, add_sections};
use urna_format_parity::{Rng, assert_parity, container_streams, records};

const DIM: usize = 16;

/// compare every zstd payload of `file`; returns how many frames.
fn check_file(name: &str, file: &[u8]) -> usize {
    let view = UrnaView::from_bytes(file).unwrap_or_else(|e| panic!("{name}: {e}"));
    let dict = view.get_section_data(SECTION_DICTIONARY).ok();
    let mut frames = 0;
    for entry in &view.section_table {
        let phys = view.get_section_data(entry.section_id).unwrap();
        let label = format!("{name} section {:#04x}", entry.section_id);
        match entry.encoding {
            SECTION_ENCODING_ZSTD => {
                assert_parity(&label, None, phys, None);
                frames += 1;
            }
            SECTION_ENCODING_TXT_STREAMS | SECTION_ENCODING_ZSTD_DICT => {
                let d = (entry.encoding == SECTION_ENCODING_ZSTD_DICT)
                    .then_some(dict)
                    .flatten();
                for (i, stream) in container_streams(phys).iter().enumerate() {
                    assert_parity(&format!("{label} stream {i}"), None, stream, d);
                    frames += 1;
                }
            }
            _ => {}
        }
        // and the section as the reader decodes it.
        view.decoded_section(entry.section_id)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
    }
    frames
}

/// The `.urna` and `.nest` files git sees (tracked, or untracked and not
/// ignored): the ones fullcheck's step cache hashes, so a new or edited
/// fixture reruns this test. An ignored local corpus is not a fixture.
fn fixtures(root: &Path) -> Vec<PathBuf> {
    let out = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .args(["--", "*.urna", "*.nest"])
        .current_dir(root)
        .output()
        .expect("git runs: the fixtures are the files git sees");
    assert!(
        out.status.success(),
        "git ls-files: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut files: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout)
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(|p| root.join(p))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files
}

/// Every fixture git sees, decoded by both decoders. A git-lfs pointer is
/// the `lfs` need (rust/clitui/tests/common/needs.rs); a fixture that does
/// not open is a failure, never a skip.
#[test]
fn tracked_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let files = fixtures(&root);
    let mut frames = 0;
    let mut opened = 0;
    for path in &files {
        let bytes = std::fs::read(path).unwrap();
        let name = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        if bytes.starts_with(b"version https://git-lfs") {
            needs::lack("lfs", &format!("{name} is a git-lfs pointer"));
            continue;
        }
        opened += 1;
        frames += check_file(&name, &bytes);
    }
    println!(
        "{opened} of {} .urna files, {frames} zstd frames compared",
        files.len()
    );
    assert!(
        opened > 0,
        "no .urna fixture found under {}",
        root.display()
    );
}

fn manifest(n: usize) -> Manifest {
    Manifest {
        embedding_model: "parity".into(),
        embedding_dim: DIM as u32,
        n_chunks: n as u64,
        model_hash: format!("sha256:{}", "0".repeat(64)),
        chunker_version: "parity-chunker/1".into(),
        capabilities: Capabilities::default(),
        ..Default::default()
    }
}

fn vectors(rng: &mut Rng, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..DIM)
                .map(|_| (rng.below(2001) as f32 - 1000.0) / 1000.0)
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            v.iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn build(texts: &[String], vecs: &[Vec<f32>]) -> Vec<u8> {
    let chunks = texts
        .iter()
        .zip(vecs)
        .enumerate()
        .map(|(i, (t, v))| ChunkInput {
            canonical_text: t.clone(),
            source_uri: format!("doc://{i}"),
            byte_start: 0,
            byte_end: t.len() as u64,
            embedding: v.clone(),
        });
    UrnaFileBuilder::new(manifest(texts.len()))
        .text_encoding(SectionEncoding::Zstd)
        .reproducible(true)
        .add_chunks(chunks)
        .build_bytes()
        .unwrap()
}

#[test]
fn built_corpora_and_a_dictionary_corpus_that_answers() {
    let mut rng = Rng::new(641);
    let texts = records(&mut rng, 600);
    let vecs = vectors(&mut rng, texts.len());
    let file = build(&texts, &vecs);
    let frames = check_file("built", &file);
    assert!(frames > 0, "the compressed preset wrote no zstd frame");

    // the same corpus with txt_streams, then with dictionary-framed text.
    let streams = encode_txt_streams(&texts).unwrap();
    let mut with_streams = Vec::new();
    let sections = vec![(
        SECTION_CHUNKS_CANONICAL,
        SECTION_ENCODING_TXT_STREAMS,
        streams,
    )];
    add_sections(&file, &mut with_streams, sections, |_| {}).unwrap();
    check_file("txt_streams", &with_streams);

    let mut unique = texts.clone();
    unique.sort_unstable();
    unique.dedup();
    let dict = train_dict(&unique).expect("the records train a dictionary");
    let framed = encode_zstd_dict(&texts, &dict).unwrap();
    let mut with_dict = Vec::new();
    let sections = vec![
        (SECTION_CHUNKS_CANONICAL, SECTION_ENCODING_ZSTD_DICT, framed),
        (SECTION_DICTIONARY, SECTION_ENCODING_RAW, dict),
    ];
    add_sections(&file, &mut with_dict, sections, |_| {}).unwrap();
    let frames = check_file("zstd_dict", &with_dict);
    assert!(frames >= texts.len(), "fewer frames than chunks");

    let path = std::env::temp_dir().join(format!("urna-zstd-parity-{}.urna", std::process::id()));
    std::fs::write(&path, &with_dict).unwrap();
    let corpus = MmapUrnaFile::open(&path).unwrap();
    let texts_read = corpus.canonical_texts().unwrap();
    assert_eq!(
        texts_read, texts,
        "the dictionary corpus reads back other text"
    );
    for i in [0, 17, 299, texts.len() - 1] {
        let hits = corpus.search(&vecs[i], 3).unwrap().hits;
        let top = corpus
            .chunk_ids()
            .iter()
            .position(|id| *id == hits[0].chunk_id)
            .unwrap();
        assert_eq!(
            texts_read[top], texts[i],
            "chunk {i} did not answer with its own text"
        );
    }
    drop(corpus);
    let _ = std::fs::remove_file(&path);
    println!(
        "[POCS] a dictionary-framed corpus of {} chunks opens and answers",
        texts.len()
    );
}
