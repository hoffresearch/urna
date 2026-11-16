//! zlib (RFC 1950) over deflate stored blocks (RFC 1951, BTYPE 00): the
//! bytes as they are, in blocks of at most 65,535, framed so that every
//! inflater reads them, with the Adler-32 the format ends on. The PNG
//! IDAT and kitty's `o=z` both take a zlib stream; neither needs it
//! compressed, so the terminal image path carries no deflate encoder.

/// Largest payload of one stored block (its LEN is a u16).
const BLOCK: usize = 65_535;

/// `data` as a zlib stream of stored blocks.
pub fn store(data: &[u8]) -> Vec<u8> {
    let blocks = data.len().div_ceil(BLOCK).max(1);
    let mut out = Vec::with_capacity(2 + blocks * 5 + data.len() + 4);
    // CMF 0x78: deflate with a 32 KiB window; FLG 0x01: no dictionary,
    // and 0x7801 is a multiple of 31, as the header check requires.
    out.extend_from_slice(&[0x78, 0x01]);
    let mut rest = data;
    loop {
        let (block, tail) = rest.split_at(rest.len().min(BLOCK));
        let last = tail.is_empty();
        // BFINAL in bit 0, BTYPE 00 (stored) in bits 1-2; a stored block's
        // header ends on the byte boundary, so the byte is BFINAL alone.
        out.push(u8::from(last));
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
        if last {
            break;
        }
        rest = tail;
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Adler-32 (RFC 1950): two sums mod 65,521, reduced every 5,552 bytes,
/// the most that cannot overflow a u32 between reductions.
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    for run in data.chunks(5552) {
        for &byte in run {
            a += u32::from(byte);
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// The bytes of a zlib stream of stored blocks (what `store` writes), or
/// None for anything else: a compressed block, a bad header, length or
/// checksum. The tests read the encoders' output back with it.
#[cfg(test)]
pub fn unstore(z: &[u8]) -> Option<Vec<u8>> {
    let (head, mut rest) = z.split_at_checked(2)?;
    if head[0] & 0x0F != 8 || ((u16::from(head[0]) << 8) | u16::from(head[1])) % 31 != 0 {
        return None;
    }
    let mut out = Vec::new();
    loop {
        let (&flags, tail) = rest.split_first()?;
        if flags & 0b110 != 0 {
            return None;
        }
        let (lens, tail) = tail.split_at_checked(4)?;
        let len = u16::from_le_bytes([lens[0], lens[1]]);
        if !len != u16::from_le_bytes([lens[2], lens[3]]) {
            return None;
        }
        let (block, tail) = tail.split_at_checked(len as usize)?;
        out.extend_from_slice(block);
        rest = tail;
        if flags & 1 == 1 {
            break;
        }
    }
    let sum = u32::from_be_bytes(rest.try_into().ok()?);
    (sum == adler32(&out)).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_matches_the_known_vectors() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
        // past one reduction run, every byte at its maximum.
        let ff = vec![0xffu8; 100_000];
        let (mut a, mut b) = (1u64, 0u64);
        for &x in &ff {
            a = (a + u64::from(x)) % 65_521;
            b = (b + a) % 65_521;
        }
        assert_eq!(adler32(&ff), ((b << 16) | a) as u32);
    }

    #[test]
    fn stores_and_reads_back_across_block_boundaries() {
        for len in [0usize, 1, 65_534, 65_535, 65_536, 200_000] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 300) as u8).collect();
            let z = store(&data);
            assert_eq!(&z[..2], &[0x78, 0x01]);
            assert_eq!(z.len(), 2 + data.len().div_ceil(BLOCK).max(1) * 5 + len + 4);
            assert_eq!(unstore(&z).as_deref(), Some(&data[..]), "len {len}");
        }
        let mut bad = store(b"abc");
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert_eq!(unstore(&bad), None);
    }
}
