//! The smallest PNG that every decoder reads: one IHDR, one IDAT holding
//! the rows unfiltered in a zlib stream (deflate through flate2, already
//! in the build), one IEND, every chunk with the CRC-32 the format
//! requires, computed here.

use std::io::Write;

use super::Rgba;

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// CRC-32 (IEEE, reflected, the PNG and zip one) over `bytes`.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// `img` as PNG bytes, RGBA 8-bit, no filter on any row.
pub fn encode(img: &Rgba) -> Vec<u8> {
    let mut raw = Vec::with_capacity((img.width as usize * 4 + 1) * img.height as usize);
    let stride = img.width as usize * 4;
    for row in img.data.chunks(stride.max(1)) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    // writing to a Vec cannot fail; an empty stream is the only outcome of
    // an error here, and a decoder reports that loudly.
    let _ = z.write_all(&raw);
    let idat = z.finish().unwrap_or_default();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&img.width.to_be_bytes());
    ihdr.extend_from_slice(&img.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    let mut out = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(&SIGNATURE);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &idat);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_known_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        // the IEND chunk's crc, the one every png carries.
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn a_two_by_two_png_has_the_header_the_chunks_and_decodes_back() {
        let img = Rgba::new(
            2,
            2,
            vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0],
        )
        .expect("2x2");
        let png = encode(&img);
        assert_eq!(&png[..8], &SIGNATURE);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..24], &[0, 0, 0, 2, 0, 0, 0, 2]);
        assert_eq!(&png[24..29], &[8, 6, 0, 0, 0]);
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
        // the idat inflates to the unfiltered rows.
        let len = u32::from_be_bytes([png[33], png[34], png[35], png[36]]) as usize;
        assert_eq!(&png[37..41], b"IDAT");
        let mut back = Vec::new();
        std::io::Read::read_to_end(
            &mut flate2::read::ZlibDecoder::new(&png[41..41 + len]),
            &mut back,
        )
        .expect("inflates");
        assert_eq!(
            back,
            vec![
                0, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 255, 0, 0, 0, 0
            ]
        );
    }

    /// the system's own reader (`sips` on macOS) decodes it: skipped where
    /// there is none.
    #[test]
    fn the_system_reader_decodes_it() {
        let Ok(sips) = std::process::Command::new("which").arg("sips").output() else {
            return;
        };
        if !sips.status.success() {
            return;
        }
        let img = Rgba::new(3, 2, vec![200; 24]).expect("3x2");
        let path = std::env::temp_dir().join(format!("urna-png-{}.png", std::process::id()));
        std::fs::write(&path, encode(&img)).expect("write");
        let out = std::process::Command::new("sips")
            .args(["-g", "pixelWidth", "-g", "pixelHeight"])
            .arg(&path)
            .output()
            .expect("sips runs");
        let _ = std::fs::remove_file(&path);
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            text.contains("pixelWidth: 3") && text.contains("pixelHeight: 2"),
            "{text}"
        );
    }
}
