//! The kitty graphics protocol (kitty, ghostty, konsole, wezterm): an
//! APC `ESC _ G <keys> ; <payload> ESC \` per command. The image goes
//! as raw RGBA (`f=32`) zlib-compressed (`o=z`), base64, in chunks of
//! 4096 with `m=1` on every chunk but the last, transmitted and placed in
//! one go (`a=T`) at the cursor, scaled into `c` x `r` cells. `q=2`
//! silences the terminal's replies, so nothing lands in the key stream.
//! `i` names the image: deleting it (`a=d,d=I`) removes its placement
//! wherever it was, which is how the previous image leaves the screen.

use std::io::Write;

use super::{Rgba, base64};

const CHUNK: usize = 4096;

pub fn encode(img: &Rgba, cols: u16, rows: u16, id: u32) -> Vec<u8> {
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    let _ = z.write_all(&img.data);
    let packed = z.finish().unwrap_or_default();
    let payload = base64::encode(&packed);
    let mut out = delete(id);
    let chunks: Vec<&[u8]> = payload.chunks(CHUNK).collect();
    let last = chunks.len().saturating_sub(1);
    for (i, chunk) in chunks.iter().enumerate() {
        out.extend_from_slice(b"\x1b_G");
        if i == 0 {
            out.extend_from_slice(
                format!(
                    "a=T,f=32,o=z,s={},v={},c={cols},r={rows},i={id},q=2",
                    img.width, img.height
                )
                .as_bytes(),
            );
        }
        if chunks.len() > 1 {
            if i != 0 {
                out.extend_from_slice(b"q=2");
            }
            out.extend_from_slice(if i == last { b",m=0" } else { b",m=1" });
        }
        out.push(b';');
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\x1b\\");
    }
    out
}

/// Remove image `id` and its placements.
pub fn delete(id: u32) -> Vec<u8> {
    format!("\x1b_Ga=d,d=I,i={id},q=2\x1b\\").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_by_two_image_is_one_delete_then_one_transmit_and_place() {
        let img = Rgba::new(
            2,
            2,
            vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255],
        )
        .expect("2x2");
        let bytes = encode(&img, 3, 2, 7);
        let text = String::from_utf8(bytes.clone()).expect("ascii");
        assert!(
            text.starts_with(
                "\x1b_Ga=d,d=I,i=7,q=2\x1b\\\x1b_Ga=T,f=32,o=z,s=2,v=2,c=3,r=2,i=7,q=2;"
            )
        );
        assert!(text.ends_with("\x1b\\"));
        assert_eq!(text.matches("\x1b_G").count(), 2);
        // the payload inflates to the pixels.
        let payload = &text[text.rfind(';').expect(";") + 1..text.len() - 2];
        let packed = decode_base64(payload);
        let mut back = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(&packed[..]), &mut back)
            .expect("inflates");
        assert_eq!(back, img.data);
    }

    #[test]
    fn a_large_image_is_chunked_with_the_continuation_flag() {
        // noise, so the zlib stream stays above one chunk.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let data: Vec<u8> = (0..64 * 64 * 4)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect();
        let img = Rgba::new(64, 64, data).expect("64x64");
        let text = String::from_utf8(encode(&img, 8, 4, 1)).expect("ascii");
        let n = text.matches("\x1b_G").count();
        assert!(n > 2, "{n} commands");
        assert_eq!(text.matches(",m=1;").count(), n - 2);
        assert_eq!(text.matches(",m=0;").count(), 1);
    }

    fn decode_base64(s: &str) -> Vec<u8> {
        let val = |c: u8| -> u32 {
            match c {
                b'A'..=b'Z' => u32::from(c - b'A'),
                b'a'..=b'z' => u32::from(c - b'a') + 26,
                b'0'..=b'9' => u32::from(c - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            }
        };
        let mut out = Vec::new();
        for q in s.as_bytes().chunks(4) {
            let n = (val(q[0]) << 18) | (val(q[1]) << 12) | (val(q[2]) << 6) | val(q[3]);
            out.push((n >> 16) as u8);
            if q[2] != b'=' {
                out.push((n >> 8) as u8);
            }
            if q[3] != b'=' {
                out.push(n as u8);
            }
        }
        out
    }
}
