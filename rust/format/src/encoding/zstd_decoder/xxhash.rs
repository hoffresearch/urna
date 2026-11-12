//! XXH64 (seed 0 in zstd), for the optional frame content checksum: the
//! frame stores the low 32 bits of the hash of its decoded content.
//! one-shot over a slice; the decoder hashes a frame once it is complete.

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn le64(b: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[..8]);
    u64::from_le_bytes(w)
}

#[inline]
fn le32(b: &[u8]) -> u64 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&b[..4]);
    u64::from(u32::from_le_bytes(w))
}

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

#[inline]
fn merge(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
}

/// the XXH64 digest of `data` with `seed`.
pub(super) fn xxh64(data: &[u8], seed: u64) -> u64 {
    let len = data.len() as u64;
    let mut rest = data;
    let mut h = if rest.len() >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        let mut stripes = rest.chunks_exact(32);
        for s in &mut stripes {
            v1 = round(v1, le64(&s[0..8]));
            v2 = round(v2, le64(&s[8..16]));
            v3 = round(v3, le64(&s[16..24]));
            v4 = round(v4, le64(&s[24..32]));
        }
        rest = stripes.remainder();
        let h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        merge(merge(merge(merge(h, v1), v2), v3), v4)
    } else {
        seed.wrapping_add(P5)
    };
    h = h.wrapping_add(len);
    let mut words = rest.chunks_exact(8);
    for w in &mut words {
        h ^= round(0, le64(w));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
    }
    rest = words.remainder();
    if rest.len() >= 4 {
        h ^= le32(rest).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        rest = &rest[4..];
    }
    for &b in rest {
        h ^= u64::from(b).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::xxh64;

    #[test]
    fn reference_vectors() {
        // published XXH64 values, seed 0.
        assert_eq!(xxh64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxh64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    }

    #[test]
    fn every_tail_path_is_deterministic() {
        // 0..=70 bytes walks the stripe loop, the 8-byte, 4-byte and 1-byte
        // tails; the parity workspace checks the values against libzstd.
        let data: Vec<u8> = (0u8..=70).collect();
        for n in 0..data.len() {
            assert_eq!(xxh64(&data[..n], 0), xxh64(&data[..n], 0));
        }
        assert_ne!(xxh64(&data[..31], 0), xxh64(&data[..32], 0));
    }
}
