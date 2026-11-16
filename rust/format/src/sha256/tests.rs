//! FIPS 180-4 / NIST CAVP vectors, the incremental API against the one
//! shot, and the instruction kernel (when this CPU has one) against the
//! portable rounds. The comparison with the `sha2` crate is in
//! `rust/format/parity`.

use super::{H0, Sha256, accel, soft};

fn hex(d: [u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn nist_vectors() {
    let cases: [(&[u8], &str); 4] = [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        // the 448-bit message: the padding needs a second block.
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        // the 896-bit message.
        (
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        ),
    ];
    for (msg, want) in cases {
        assert_eq!(hex(Sha256::digest(msg)), want, "{}", msg.len());
    }
}

#[test]
#[cfg_attr(miri, ignore)] // a million bytes through the portable rounds: too slow under an interpreter
fn a_million_a() {
    let msg = vec![b'a'; 1_000_000];
    let want = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    assert_eq!(hex(Sha256::digest(&msg)), want);
    // the same million fed in uneven pieces, across every block boundary.
    let mut h = Sha256::new();
    let mut rest = &msg[..];
    let mut step = 1;
    while !rest.is_empty() {
        let n = step.min(rest.len());
        h.update(&rest[..n]);
        rest = &rest[n..];
        step = step % 131 + 7;
    }
    assert_eq!(hex(h.finalize()), want);
}

#[test]
fn streaming_equals_one_shot_at_every_split() {
    let msg: Vec<u8> = (0..300u32).map(|i| (i * 31 + 7) as u8).collect();
    for len in [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 300] {
        let whole = Sha256::digest(&msg[..len]);
        for split in 0..=len {
            let mut h = Sha256::new();
            h.update(&msg[..split]);
            h.update(&msg[split..len]);
            assert_eq!(h.finalize(), whole, "len {len} split {split}");
        }
    }
}

#[test]
fn a_clone_finalizes_where_it_was_taken() {
    let mut h = Sha256::new();
    h.update(b"ab");
    let snapshot = h.clone();
    h.update(b"c");
    assert_eq!(snapshot.finalize(), Sha256::digest(b"ab"));
    assert_eq!(h.finalize(), Sha256::digest(b"abc"));
}

#[test]
fn the_instruction_kernel_matches_the_portable_rounds() {
    let msg: Vec<u8> = (0..64 * 40u32).map(|i| (i ^ (i >> 5)) as u8).collect();
    for blocks in [1, 2, 3, 40] {
        let mut ours = H0;
        let mut reference = H0;
        soft::compress(&mut reference, &msg[..64 * blocks]);
        if accel::compress(&mut ours, &msg[..64 * blocks]) {
            assert_eq!(ours, reference, "{blocks} blocks");
        } else {
            // no SHA instructions here (or miri): nothing to compare, and
            // the state must be untouched for the portable path.
            assert_eq!(ours, H0);
        }
    }
}
