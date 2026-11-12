//! float16 conversions for the `encoding=2` embeddings path. The
//! runtime never accumulates in f16 - quantize at write time, decode
//! lane-by-lane into f32 at read time.
//!
//! The conversion is IEEE 754 binary16 done on the bits: f32 to f16 rounds
//! to nearest, ties to even (what F16C and ARM's `fcvt` do), overflow goes
//! to infinity, values below half the smallest subnormal go to a signed
//! zero, and a NaN keeps its sign and top payload bits with the quiet bit
//! set. f16 to f32 is exact for every one of the 65,536 patterns.
//! `rust/format/parity` holds both directions to the `half` crate.

/// One IEEE 754 binary16 value, stored as its bits. The on-disk f16 of
/// the embeddings and the int4 group scales is exactly these bits, little
/// endian.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F16(u16);

impl F16 {
    /// The quiet NaN F16C and ARM produce (`0x7E00`).
    pub const NAN: F16 = F16(0x7E00);
    /// Positive infinity (`0x7C00`).
    pub const INFINITY: F16 = F16(0x7C00);

    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// Round `v` to the nearest f16, ties to even.
    #[inline]
    pub fn from_f32(v: f32) -> Self {
        let x = v.to_bits();
        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xFF) as i32;
        let man = x & 0x007F_FFFF;
        if exp == 0xFF {
            // infinity keeps a zero mantissa; a NaN keeps its top ten
            // payload bits and is quieted, as the hardware conversions do.
            let nan = if man == 0 {
                0
            } else {
                0x0200 | (man >> 13) as u16
            };
            return Self(sign | 0x7C00 | nan);
        }
        // rebias from f32's 127 to f16's 15.
        let e = exp - 112;
        if e >= 0x1F {
            return Self(sign | 0x7C00);
        }
        if e <= 0 {
            // an f16 subnormal (or zero): the 24-bit significand, hidden
            // bit included, shifted right by 14 - e places. below e = -10
            // even the round bit falls off, so the result is a signed zero.
            if e < -10 {
                return Self(sign);
            }
            let m = man | 0x0080_0000;
            let shift = (14 - e) as u32;
            let mut h = m >> shift;
            let rest = m & ((1 << shift) - 1);
            let tie = 1 << (shift - 1);
            if rest > tie || (rest == tie && h & 1 == 1) {
                // a carry out of the mantissa lands in the exponent field,
                // which is the smallest normal: still the right value.
                h += 1;
            }
            return Self(sign | h as u16);
        }
        let mut h = ((e as u32) << 10) | (man >> 13);
        let rest = man & 0x1FFF;
        if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
            // a carry may roll into the exponent, up to infinity at the
            // top of the range, which is round to nearest's answer too.
            h += 1;
        }
        Self(sign | h as u16)
    }

    /// The exact f32 of this f16.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1F;
        let man = h & 0x03FF;
        let bits = if exp == 0x1F {
            if man == 0 {
                sign | 0x7F80_0000
            } else {
                sign | 0x7FC0_0000 | (man << 13)
            }
        } else if exp != 0 {
            sign | ((exp + 112) << 23) | (man << 13)
        } else if man == 0 {
            sign
        } else {
            // subnormal: man * 2^-24. normalize so the leading one sits on
            // bit 10 (the hidden bit), and lower the exponent to match.
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x03FF) << 13
        };
        f32::from_bits(bits)
    }
}

/// Convert an L2-normalized float32 embedding to float16, returning the
/// little-endian byte representation. Not a checked NaN/Inf path -
/// callers must validate inputs upstream.
pub fn f32_to_f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&F16::from_f32(v).to_le_bytes());
    }
    out
}

/// Decode a float16 byte slice into f32 values, accumulating in f32
/// for downstream dot-product accuracy.
#[inline]
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(F16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::F16;

    #[test]
    fn known_values() {
        for (v, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (0.5, 0x3800),
            (65504.0, 0x7BFF),
            (65520.0, 0x7C00), // the tie above the largest finite rounds up to infinity
            (f32::INFINITY, 0x7C00),
            (f32::NEG_INFINITY, 0xFC00),
            (f32::from_bits(103 << 23), 0x0001), // 2^-24, the smallest subnormal
            (f32::from_bits(102 << 23), 0x0000), // 2^-25, the tie to zero (even)
            (f32::from_bits(113 << 23), 0x0400), // 2^-14, the smallest normal
        ] {
            assert_eq!(F16::from_f32(v).to_bits(), bits, "{v}");
        }
        assert_eq!(F16::from_f32(f32::NAN).to_bits(), 0x7E00);
        assert!(F16::NAN.to_f32().is_nan());
        assert_eq!(F16::INFINITY.to_f32(), f32::INFINITY);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 65,536 conversions each way: too slow under an interpreter
    fn every_finite_f16_round_trips_through_f32() {
        for bits in 0..=u16::MAX {
            let h = F16::from_bits(bits);
            let f = h.to_f32();
            if f.is_nan() {
                assert_eq!(bits & 0x7C00, 0x7C00);
                continue;
            }
            assert_eq!(F16::from_f32(f).to_bits(), bits, "{bits:#06x}");
        }
    }

    #[test]
    fn ties_go_to_even() {
        // 1 + 2^-11 sits halfway between 1.0 (even) and 1 + 2^-10 (odd).
        assert_eq!(F16::from_f32(1.0 + 2f32.powi(-11)).to_bits(), 0x3C00);
        // 1 + 3 * 2^-11 sits halfway between 0x3C01 (odd) and 0x3C02 (even).
        assert_eq!(F16::from_f32(1.0 + 3.0 * 2f32.powi(-11)).to_bits(), 0x3C02);
    }
}
