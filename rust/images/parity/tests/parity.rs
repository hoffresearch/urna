//! Every AVIF blob of a card corpus decoded by this crate and by
//! `avif-decode` (the decoder the explorer used before), pixel against
//! pixel. The corpus is the file named by `URNA_CARDS_CORPUS`; without it
//! the test passes doing nothing and says so. `URNA_PARITY_LIMIT` caps
//! how many blobs are compared.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test: a failure is the finding"
)]

use std::time::Instant;

fn reference(bytes: &[u8]) -> (u32, u32, Vec<u8>) {
    let img = avif_decode::Decoder::from_avif(bytes)
        .and_then(|d| d.to_image())
        .unwrap_or_else(|e| panic!("avif-decode failed: {e}"));
    let to8 = |v: u16| (v >> 8) as u8;
    match img {
        avif_decode::Image::Rgb8(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels().flat_map(|p| [p.r, p.g, p.b, 255]).collect(),
        ),
        avif_decode::Image::Rgba8(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels().flat_map(|p| [p.r, p.g, p.b, p.a]).collect(),
        ),
        avif_decode::Image::Rgb16(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels()
                .flat_map(|p| [to8(p.r), to8(p.g), to8(p.b), 255])
                .collect(),
        ),
        avif_decode::Image::Rgba16(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels()
                .flat_map(|p| [to8(p.r), to8(p.g), to8(p.b), to8(p.a)])
                .collect(),
        ),
        avif_decode::Image::Gray8(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels()
                .flat_map(|p| [p.value(), p.value(), p.value(), 255])
                .collect(),
        ),
        avif_decode::Image::Gray16(i) => (
            i.width() as u32,
            i.height() as u32,
            i.pixels()
                .flat_map(|p| {
                    let v = to8(p.value());
                    [v, v, v, 255]
                })
                .collect(),
        ),
    }
}

#[test]
fn every_blob_of_the_corpus_decodes_like_avif_decode() {
    let Some(path) = std::env::var_os("URNA_CARDS_CORPUS") else {
        eprintln!("URNA_CARDS_CORPUS unset: parity skipped");
        return;
    };
    let limit: usize = std::env::var("URNA_PARITY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let rt = urna_engine::MmapUrnaFile::open(std::path::Path::new(&path)).expect("corpus opens");
    let refs = rt.blob_refs().expect("the corpus has blobs");
    let avif: Vec<usize> = refs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.inlined && r.original_uri.ends_with(".avif"))
        .map(|(i, _)| i)
        .take(limit)
        .collect();
    let (mut equal, mut off_by_one, mut worse) = (0usize, 0usize, 0usize);
    let (mut ours_ms, mut theirs_ms) = (0f64, 0f64);
    let mut max_diff = 0u8;
    let mut first_diff: Option<(usize, usize, u8)> = None;
    for &i in &avif {
        let bytes = rt.blob_bytes(i).expect("blob bytes");
        let t = Instant::now();
        let ours = urna_images::decode(bytes).unwrap_or_else(|e| panic!("blob {i}: {e}"));
        ours_ms += t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let (w, h, theirs) = reference(bytes);
        theirs_ms += t.elapsed().as_secs_f64() * 1e3;
        assert_eq!((ours.width, ours.height), (w, h), "blob {i}: size");
        assert_eq!(ours.rgba.len(), theirs.len(), "blob {i}: pixel count");
        let d = ours
            .rgba
            .iter()
            .zip(&theirs)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        max_diff = max_diff.max(d);
        match d {
            0 => equal += 1,
            1 => off_by_one += 1,
            _ => {
                worse += 1;
                if first_diff.is_none() {
                    let px = ours
                        .rgba
                        .iter()
                        .zip(&theirs)
                        .position(|(a, b)| a.abs_diff(*b) == d)
                        .unwrap_or(0);
                    first_diff = Some((i, px, d));
                }
            }
        }
    }
    let n = avif.len().max(1) as f64;
    eprintln!(
        "parity over {} blobs: {equal} identical, {off_by_one} within 1, {worse} further (max diff {max_diff}); \
         ours {:.2} ms/image, avif-decode {:.2} ms/image; first larger diff {first_diff:?}",
        avif.len(),
        ours_ms / n,
        theirs_ms / n
    );
    // the AV1 planes are exact (tests/planes.rs); the RGB conversion is
    // ours and avif-decode's is the `yuv` crate's, two roundings of the
    // same matrix, so a pixel may land two steps away.
    assert!(max_diff <= 2, "pixels further than 2 apart: {first_diff:?}");
}
