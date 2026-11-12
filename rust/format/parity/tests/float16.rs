//! urna_format::F16 against the `half` crate it replaced: every f16 bit
//! pattern widened to f32, and f32 narrowed to f16 on every rounding
//! boundary of the f16 grid plus a large seeded sample.
//!
//! Bits are compared, NaN included: `half` takes the F16C or ARM `fcvt`
//! instruction when the CPU has one, so on the CI runners this also holds
//! our software conversion to the hardware's answer.
//!
//! `URNA_PARITY_EXHAUSTIVE=1` narrows all 2^32 f32 patterns instead of the
//! sample (about a minute in release).

use urna_format::F16;

fn theirs_narrow(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn ours_narrow(v: f32) -> u16 {
    F16::from_f32(v).to_bits()
}

fn check_narrow(bits: u32) {
    let v = f32::from_bits(bits);
    assert_eq!(
        ours_narrow(v),
        theirs_narrow(v),
        "f32 {bits:#010x} ({v:e}) narrows differently"
    );
}

/// splitmix64: a seeded, dependency-free source of test inputs.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn every_f16_widens_to_the_same_f32_bits() {
    for bits in 0..=u16::MAX {
        let ours = F16::from_bits(bits).to_f32().to_bits();
        let theirs = half::f16::from_bits(bits).to_f32().to_bits();
        assert_eq!(ours, theirs, "f16 {bits:#06x} widens differently");
    }
}

#[test]
fn narrowing_agrees_on_every_rounding_boundary() {
    // each f16 value, and the f32 halfway to its upper neighbour, with a
    // few f32 ulps on either side: the exact points where round to
    // nearest, ties to even, decides.
    for bits in 0..0x7C00u16 {
        for sign in [0u16, 0x8000] {
            let lo = F16::from_bits(bits | sign).to_f32();
            let hi = F16::from_bits((bits + 1) | sign).to_f32();
            let mid = (lo as f64 + hi as f64) / 2.0;
            for centre in [lo.to_bits(), (mid as f32).to_bits()] {
                for d in 0..=4u32 {
                    check_narrow(centre.wrapping_add(d));
                    check_narrow(centre.wrapping_sub(d));
                }
            }
        }
    }
}

#[test]
fn narrowing_agrees_on_specials() {
    for bits in [
        0x0000_0000u32,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0x7F80_0001,
        0x7FBF_FFFF,
        0xFFFF_FFFF,
        0x7FFF_E000,
        0x0000_0001,
        0x807F_FFFF,
        0x477F_F000,
        0x477F_EFFF,
        0x3380_0000,
        0x3380_0001,
        0x337F_FFFF,
    ] {
        check_narrow(bits);
    }
    // every NaN payload class: the top ten mantissa bits are kept.
    for top in 0..1024u32 {
        check_narrow(0x7F80_0000 | (top << 13) | 1);
        check_narrow(0xFF80_0000 | (top << 13) | 0x1FFF);
    }
}

#[test]
fn narrowing_agrees_on_a_seeded_sample() {
    if std::env::var("URNA_PARITY_EXHAUSTIVE").is_ok_and(|v| v == "1") {
        for bits in 0..=u32::MAX {
            check_narrow(bits);
        }
        return;
    }
    let mut state = 0x5EED_F16F_16F1_6F16u64;
    for _ in 0..16_000_000 {
        let r = splitmix(&mut state);
        check_narrow(r as u32);
        // half of the draws inside the f16 range (|x| < 2^17), where the
        // rounding happens, instead of uniform over all exponents.
        let near = (r >> 32) as u32;
        check_narrow((near & 0x807F_FFFF) | (((near >> 23) % 44 + 100) << 23));
    }
}
