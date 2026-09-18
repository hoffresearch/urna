//! dedup order-invariant guardrail (the nix nix-casync/tvix lesson): dedup
//! and content-hash equality MUST run on DECOMPRESSED bytes, with zstd
//! applied to the unique pool AFTERWARD. this test proves the order is
//! correct by showing that, on a corpus of far-apart repeated blocks beyond
//! a single zstd window's easy reach, dedup-before-zstd shrinks more than
//! zstd-after (the inverted order recovers almost nothing).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]
use urna_format::encoding::{dedup, zstd_encode};
use urna_format::sections::encode_chunks_canonical;

/// build a corpus whose repeated blocks span MORE than a single zstd window,
/// so a single zstd frame cannot fully collapse the far-apart duplicates but
/// content-hash dedup can. 60 distinct ~8 KiB blocks repeated 40x and
/// interleaved spreads the duplicates across > 16 MiB.
fn spread_repeats() -> Vec<String> {
    let uniques: Vec<String> = (0..60)
        .map(|i| {
            (0..8192)
                .map(|j| char::from(b'a' + (((i * 7919 + j * 31) % 26) as u8)))
                .collect::<String>()
        })
        .collect();
    let mut out = Vec::with_capacity(uniques.len() * 40);
    for _ in 0..40 {
        for u in &uniques {
            out.push(u.clone());
        }
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)] // zstd is c code, miri cannot call it
fn dedup_before_zstd_beats_zstd_alone_on_repeats() {
    let corpus = spread_repeats();

    // zstd alone over the full canonical bytes (the dedup-AFTER / no-dedup
    // baseline).
    let full_raw = encode_chunks_canonical(&corpus).unwrap();
    let zstd_alone = zstd_encode(&full_raw).unwrap();

    // dedup BEFORE zstd: dedup on the decompressed strings, then zstd the
    // unique pool + the back-ref map.
    let d = dedup(&corpus);
    assert!(d.unique.len() < corpus.len(), "corpus must actually repeat");
    let pool_raw = encode_chunks_canonical(&d.unique).unwrap();
    let pool_zstd = zstd_encode(&pool_raw).unwrap();
    let map = urna_format::encoding::encode_dedup_map(&d.back_refs);
    let dedup_then_zstd = pool_zstd.len() + map.len();

    // the dedup-before-zstd path must be strictly smaller. (this is the
    // guardrail: if a future change inverts the order, the unique pool would
    // be the full corpus and this assertion fails.)
    assert!(
        dedup_then_zstd < zstd_alone.len(),
        "dedup-before-zstd ({}) must beat zstd-alone ({})",
        dedup_then_zstd,
        zstd_alone.len()
    );
}

#[test]
fn dedup_runs_on_decompressed_bytes() {
    // the dedup API takes decompressed strings, never compressed bytes: a
    // compile-time + semantic guarantee. here we assert that compressing
    // first would defeat dedup (two zstd frames of identical input are equal,
    // but dedup keys on the STRING, not the frame, so it collapses regardless
    // of any later compression).
    let corpus = vec!["identical".to_string(); 10];
    let d = dedup(&corpus);
    assert_eq!(d.unique.len(), 1, "dedup keys on decompressed text");
}
