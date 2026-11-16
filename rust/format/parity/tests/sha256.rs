//! urna_format::Sha256 against the `sha2` crate it replaced: seeded
//! messages of every length from 0 to 10,000 bytes, a few multi-megabyte
//! ones, and the same messages fed in random pieces. Both pick the CPU's
//! SHA instructions when present, so on the CI runners (SHA-NI on x86_64,
//! the SHA2 extension on aarch64) this holds our kernel to theirs.

use sha2::Digest;
use urna_format::Sha256;

/// splitmix64: a seeded, dependency-free source of test inputs.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn bytes(state: &mut u64, len: usize) -> Vec<u8> {
    (0..len).map(|_| splitmix(state) as u8).collect()
}

fn theirs(msg: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&sha2::Sha256::digest(msg));
    out
}

/// `msg` fed to ours in random pieces of up to `max` bytes.
fn ours_in_pieces(state: &mut u64, msg: &[u8], max: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut rest = msg;
    while !rest.is_empty() {
        let n = (splitmix(state) as usize % max + 1).min(rest.len());
        h.update(&rest[..n]);
        rest = &rest[n..];
    }
    h.finalize()
}

#[test]
fn every_length_up_to_ten_thousand() {
    let mut state = 0x5EED_5A25_6000_0001u64;
    for len in 0..=10_000 {
        let msg = bytes(&mut state, len);
        let want = theirs(&msg);
        assert_eq!(Sha256::digest(&msg), want, "len {len}");
        assert_eq!(
            ours_in_pieces(&mut state, &msg, 150),
            want,
            "len {len} in pieces"
        );
    }
}

#[test]
fn a_few_megabytes() {
    let mut state = 0x5EED_5A25_6000_0002u64;
    for len in [1 << 20, 3 * (1 << 20) + 17, 8 * (1 << 20) - 1] {
        let msg = bytes(&mut state, len);
        let want = theirs(&msg);
        assert_eq!(Sha256::digest(&msg), want, "len {len}");
        assert_eq!(
            ours_in_pieces(&mut state, &msg, 1 << 16),
            want,
            "len {len} in pieces"
        );
    }
}
