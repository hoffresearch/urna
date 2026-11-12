//! SHA-256 (FIPS 180-4), the hash of every integrity surface of the
//! format: the header and section checksums, `file_hash`, `content_hash`
//! and `chunk_id`.
//!
//! The compression function runs on the CPU's SHA instructions when it has
//! them (SHA-NI on x86_64, the ARMv8 SHA2 extension on aarch64, both
//! detected at run time) and on the portable rounds in `soft` otherwise;
//! the three are held to each other by the tests below and to the `sha2`
//! crate by `rust/format/parity`. The instruction kernels are the only
//! `unsafe` of this crate, confined to `accel`.

// the only `unsafe` of the crate (lib.rs denies it everywhere else).
#[allow(unsafe_code)]
mod accel;
mod soft;

/// Initial hash value, H(0) of FIPS 180-4 section 5.3.3.
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Round constants, K of FIPS 180-4 section 4.2.2. A static, not a const:
/// the instruction kernels load from its address.
pub(crate) static K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Run the compression function over `blocks` (a multiple of 64 bytes).
fn compress(state: &mut [u32; 8], blocks: &[u8]) {
    debug_assert_eq!(blocks.len() % 64, 0);
    if !accel::compress(state, blocks) {
        soft::compress(state, blocks);
    }
}

/// An incremental SHA-256: `update` any number of times, then `finalize`.
/// `Sha256::digest` is the one-shot form.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    /// the bytes of a block not yet compressed, `buffered` of them.
    block: [u8; 64],
    buffered: usize,
    /// message length so far, in bytes.
    length: u64,
}

impl std::fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // the length only: the state is a function of the hashed bytes.
        f.debug_struct("Sha256")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: H0,
            block: [0; 64],
            buffered: 0,
            length: 0,
        }
    }

    /// The digest of `data` in one call.
    pub fn digest(data: impl AsRef<[u8]>) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    pub fn update(&mut self, data: impl AsRef<[u8]>) {
        let mut data = data.as_ref();
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.block[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered < 64 {
                return;
            }
            compress(&mut self.state, &self.block);
            self.buffered = 0;
        }
        // whole blocks straight from the input, the remainder buffered.
        let whole = data.len() - data.len() % 64;
        if whole > 0 {
            compress(&mut self.state, &data[..whole]);
        }
        let rest = &data[whole..];
        self.block[..rest.len()].copy_from_slice(rest);
        self.buffered = rest.len();
    }

    /// Pad (a one bit, zeros, the bit length as a big-endian u64) and
    /// return the digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.length.wrapping_mul(8);
        let mut tail = [0u8; 128];
        tail[..self.buffered].copy_from_slice(&self.block[..self.buffered]);
        tail[self.buffered] = 0x80;
        // the length needs 8 bytes after the one bit: one block or two.
        let end = if self.buffered < 56 { 64 } else { 128 };
        tail[end - 8..end].copy_from_slice(&bits.to_be_bytes());
        compress(&mut self.state, &tail[..end]);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests;
