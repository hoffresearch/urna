//! Lowercase hex for digests and checksums: `sha256:<64 hex>` ids, section
//! checksums in `inspect`, model fingerprints. Two loops, no table beyond
//! the sixteen digits, so the format crate needs no hex dependency.

use crate::error::UrnaError;

const DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Lowercase hex of `bytes`, two digits per byte.
pub fn to_hex(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(char::from(DIGITS[usize::from(b >> 4)]));
        out.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    out
}

/// The bytes of a hex string, either case. An odd length or a character
/// outside `0-9a-fA-F` is `InvalidInput` naming it; callers that need the
/// canonical lowercase form compare `to_hex` of the result with the input.
pub fn from_hex(text: impl AsRef<[u8]>) -> crate::Result<Vec<u8>> {
    let text = text.as_ref();
    if text.len() % 2 != 0 {
        return Err(UrnaError::InvalidInput(format!(
            "hex: odd number of digits ({})",
            text.len()
        )));
    }
    let nibble = |c: u8, at: usize| match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(UrnaError::InvalidInput(format!(
            "hex: invalid character {:?} at position {at}",
            char::from(c)
        ))),
    };
    text.chunks_exact(2)
        .enumerate()
        .map(|(i, pair)| -> crate::Result<u8> {
            Ok((nibble(pair[0], 2 * i)? << 4) | nibble(pair[1], 2 * i + 1)?)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_round_trips() {
        let all: Vec<u8> = (0..=255).collect();
        let text = to_hex(&all);
        assert_eq!(text.len(), 512);
        assert!(text.starts_with("000102") && text.ends_with("fdfeff"));
        assert_eq!(from_hex(&text).unwrap(), all);
        assert_eq!(from_hex(text.to_uppercase()).unwrap(), all);
    }

    #[test]
    fn a_known_digest_reads_as_written() {
        // sha256 of the empty string.
        let hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let bytes = from_hex(hex).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes[0], 0xe3);
        assert_eq!(to_hex(&bytes), hex);
        assert_eq!(to_hex([0u8; 0]), "");
        assert_eq!(from_hex("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn bad_input_is_named_not_guessed() {
        let odd = from_hex("abc").unwrap_err().to_string();
        assert!(odd.contains("odd number of digits (3)"), "{odd}");
        let bad = from_hex("0g").unwrap_err().to_string();
        assert!(bad.contains("invalid character 'g' at position 1"), "{bad}");
        assert!(from_hex("é0").is_err());
    }
}
