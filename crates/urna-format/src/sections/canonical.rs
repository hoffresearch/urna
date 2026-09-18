//! `chunks_canonical` section (`SECTION_CHUNKS_CANONICAL = 0x02`).
//! One length-prefixed UTF-8 string per chunk — the canonical text the
//! `chunk_id` was derived from.

use super::codec::{Cursor, read_prefix, write_lp_str, write_prefix};
use crate::error::UrnaError;
use crate::layout::SECTION_CHUNKS_CANONICAL;

pub fn encode_chunks_canonical(texts: &[String]) -> crate::Result<Vec<u8>> {
    let mut buf = Vec::new();
    write_prefix(&mut buf, texts.len() as u64);
    for t in texts {
        write_lp_str(&mut buf, t)?;
    }
    Ok(buf)
}

pub fn decode_chunks_canonical(data: &[u8], expected_count: usize) -> crate::Result<Vec<String>> {
    let mut c = Cursor::new(data, SECTION_CHUNKS_CANONICAL);
    let count = read_prefix(&mut c)? as usize;
    if count != expected_count {
        return Err(UrnaError::SectionCountMismatch {
            section_id: SECTION_CHUNKS_CANONICAL,
            expected: expected_count,
            got: count,
        });
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(c.read_lp_str()?);
    }
    c.finish()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let texts = vec!["primeiro".to_string(), "segundo".to_string()];
        let bytes = encode_chunks_canonical(&texts).unwrap();
        let back = decode_chunks_canonical(&bytes, 2).unwrap();
        assert_eq!(texts, back);
    }
}
