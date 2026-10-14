//! Byte-level mutation engine for the mutation-fuzz tests: a seeded
//! xorshift rng, one mutation step that mirrors what libFuzzer's default
//! mutators do (bit flips, byte sets, zero runs, integer specials at
//! aligned positions, truncation, splices), and `reseal`, which recomputes
//! the header checksum, every section checksum and the footer hash so a
//! mutated file passes the integrity layer and reaches the decoders.

use sha2::{Digest, Sha256};
use urna_format::layout::{
    SectionEntry, URNA_FOOTER_SIZE, URNA_HEADER_SIZE, URNA_MAGIC, URNA_SECTION_ENTRY_SIZE,
    UrnaFooter, UrnaHeader,
};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.u64() % n as u64) as usize
        }
    }
    pub fn f32(&mut self) -> f32 {
        (self.u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(bytes));
    out
}

const SPECIALS_U32: [u32; 6] = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF];
const SPECIALS_U64: [u64; 6] = [
    0,
    1,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    0xFFFF_FFFF_FFFF_FFFE,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// One to four mutation steps over a copy of `base`.
pub fn mutate(base: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut b = base.to_vec();
    let steps = 1 + rng.below(4);
    for _ in 0..steps {
        if b.is_empty() {
            b.push(rng.u64() as u8);
            continue;
        }
        match rng.below(9) {
            0 => {
                let i = rng.below(b.len());
                b[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(b.len());
                b[i] = rng.u64() as u8;
            }
            2 => {
                let i = rng.below(b.len());
                let run = (1 + rng.below(64)).min(b.len() - i);
                b[i..i + run].fill(0);
            }
            3 => {
                let i = rng.below(b.len());
                let run = (1 + rng.below(64)).min(b.len() - i);
                b[i..i + run].fill(0xFF);
            }
            4 if b.len() >= 4 => {
                let i = rng.below(b.len() / 4) * 4;
                let v = SPECIALS_U32[rng.below(SPECIALS_U32.len())];
                b[i..i + 4].copy_from_slice(&v.to_le_bytes());
            }
            5 if b.len() >= 8 => {
                let i = rng.below(b.len() / 8) * 8;
                let v = if rng.below(2) == 0 {
                    SPECIALS_U64[rng.below(SPECIALS_U64.len())]
                } else {
                    b.len() as u64 + rng.below(3) as u64 - 1
                };
                b[i..i + 8].copy_from_slice(&v.to_le_bytes());
            }
            6 => {
                let cut = 1 + rng.below(b.len().max(2) / 2);
                b.truncate(b.len() - cut);
            }
            7 if b.len() >= 16 => {
                let len = 1 + rng.below(b.len() / 4);
                let src = rng.below(b.len() - len);
                let dst = rng.below(b.len() - len);
                let window = b[src..src + len].to_vec();
                b[dst..dst + len].copy_from_slice(&window);
            }
            _ => {
                let i = rng.below(b.len() + 1);
                let extra: Vec<u8> = (0..1 + rng.below(32)).map(|_| rng.u64() as u8).collect();
                b.splice(i..i, extra);
            }
        }
    }
    b
}

/// Recompute the header checksum, every in-bounds section checksum and the
/// footer file hash, so a mutation reaches the decoders instead of dying at
/// the integrity layer. Files too short to hold a header + footer, or with a
/// wrong magic, are left as they are (the reader must reject those anyway).
pub fn reseal(b: &mut [u8]) {
    if b.len() < URNA_HEADER_SIZE + URNA_FOOTER_SIZE || &b[..4] != URNA_MAGIC {
        return;
    }
    let mut header = UrnaHeader::default();
    header
        .as_bytes_mut()
        .copy_from_slice(&b[..URNA_HEADER_SIZE]);
    header.file_size = b.len() as u64;
    header.compute_checksum();
    b[..URNA_HEADER_SIZE].copy_from_slice(header.as_bytes());

    let body_end = b.len() - URNA_FOOTER_SIZE;
    let table_off = header.section_table_offset as usize;
    let count = header.section_table_count as usize;
    for i in 0..count {
        let Some(at) = table_off.checked_add(i * URNA_SECTION_ENTRY_SIZE) else {
            break;
        };
        if at
            .checked_add(URNA_SECTION_ENTRY_SIZE)
            .is_none_or(|end| end > body_end)
        {
            break;
        }
        let mut e = SectionEntry::new(0, 0, 0);
        e.as_bytes_mut()
            .copy_from_slice(&b[at..at + URNA_SECTION_ENTRY_SIZE]);
        let start = e.offset as usize;
        let Some(end) = start.checked_add(e.size as usize) else {
            continue;
        };
        if end > body_end {
            continue;
        }
        let payload = b[start..end].to_vec();
        e.compute_checksum(&payload);
        b[at..at + URNA_SECTION_ENTRY_SIZE].copy_from_slice(e.as_bytes());
    }
    let hash = UrnaFooter::compute_file_hash(&b[..body_end]);
    let footer = UrnaFooter::new(hash);
    b[body_end..].copy_from_slice(footer.as_bytes());
}
