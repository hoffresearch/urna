//! Shared helpers of the zstd parity tests (`tests/`): deterministic inputs,
//! libzstd as the reference decoder, our decoder, and the comparison. The
//! crate exists for its tests, which hold urna-format's own zstd decoder to
//! libzstd byte for byte.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test support: a failure is the finding"
)]

use urna_format::encoding::{IntpackReader, pack_u64s, zstd_decompress, zstd_decompress_with_dict};
use zstd::zstd_safe::DParameter;

/// the output limit both decoders run under.
pub const CAP: usize = 1 << 28;

/// xorshift64*: deterministic inputs without a dependency.
#[derive(Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n).map(|_| rng.next_u64() as u8).collect()
}

const WORDS: &[&str] = &[
    "o",
    "arquivo",
    "é",
    "o",
    "banco",
    "de",
    "dados",
    "the",
    "file",
    "is",
    "database",
    "citação",
    "verificável",
    "urna",
    "search",
    "offline",
    "vector",
    "embedding",
    "hash",
    "conteúdo",
    "seção",
    "texto",
    "zstd",
    "dictionary",
    "frame",
    "block",
    "literal",
    "sequence",
    "offset",
    "match",
    "1984",
    "2026",
    "42",
    "não",
    "sim",
    "corpus",
    "chunk",
    "índice",
    "bm25",
    "hnsw",
    "cosine",
];

/// text-like bytes: words of a small mixed-language vocabulary with
/// punctuation, numbers and line breaks, so matches and Huffman literals
/// both show up.
pub fn text(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n + 16);
    while out.len() < n {
        out.extend_from_slice(WORDS[rng.below(WORDS.len())].as_bytes());
        out.push(match rng.below(20) {
            0 => b'\n',
            1 => b',',
            2 => b'.',
            _ => b' ',
        });
        if rng.below(50) == 0 {
            out.extend_from_slice(rng.next_u64().to_string().as_bytes());
        }
    }
    out.truncate(n);
    out
}

/// short structured records with shared boilerplate, the shape a trained
/// dictionary is for.
pub fn records(rng: &mut Rng, count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            let body = String::from_utf8_lossy(&text(rng, 20 + rng.below(200))).into_owned();
            format!(
                "registro {i} | status: {} | seção {} | autor: equipe urna\n{body}",
                ["ativo", "arquivado", "rascunho"][rng.below(3)],
                rng.below(40)
            )
        })
        .collect()
}

/// libzstd's decode, the reference (a prepared dictionary, as the format
/// crate used before, and the largest window).
pub fn reference(frame: &[u8], dict: Option<&[u8]>) -> Vec<u8> {
    let mut d = match dict {
        Some(dict) => zstd::bulk::Decompressor::with_dictionary(dict).unwrap(),
        None => zstd::bulk::Decompressor::new().unwrap(),
    };
    d.set_parameter(DParameter::WindowLogMax(31)).unwrap();
    d.decompress(frame, CAP)
        .unwrap_or_else(|e| panic!("libzstd refused the frame: {e}"))
}

/// our decode.
pub fn ours(frame: &[u8], dict: Option<&[u8]>) -> Result<Vec<u8>, String> {
    match dict {
        Some(dict) => zstd_decompress_with_dict(frame, dict, CAP),
        None => zstd_decompress(frame, CAP),
    }
    .map_err(|e| e.to_string())
}

/// both decoders give `original` (when known) and the same bytes.
pub fn assert_parity(name: &str, original: Option<&[u8]>, frame: &[u8], dict: Option<&[u8]>) {
    let theirs = reference(frame, dict);
    let mine = ours(frame, dict).unwrap_or_else(|e| panic!("{name}: our decoder refused: {e}"));
    assert!(
        mine == theirs,
        "{name}: decoders disagree ({} vs {} bytes)",
        mine.len(),
        theirs.len()
    );
    if let Some(original) = original {
        assert!(mine == original, "{name}: not the original bytes");
    }
}

/// the per-chunk streams of a txt_streams container (V1 plain zstd, V2
/// dictionary-framed): kind byte, u64 count, intpack offset table, streams.
pub fn container_streams(bytes: &[u8]) -> Vec<&[u8]> {
    let rest = &bytes[9..];
    let reader = IntpackReader::parse(rest).unwrap();
    let offsets: Vec<u64> = (0..reader.len()).map(|i| reader.get(i).unwrap()).collect();
    let streams = &rest[pack_u64s(&offsets).len()..];
    offsets
        .windows(2)
        .map(|w| &streams[w[0] as usize..w[1] as usize])
        .collect()
}
