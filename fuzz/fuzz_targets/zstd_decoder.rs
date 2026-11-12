#![no_main]
//! urna-format's own zstd decoder against libzstd (#645). the first byte
//! picks the mode:
//!
//! - arbitrary bytes, no dictionary: ours must never panic, and when both
//!   decoders accept the input they must agree byte for byte;
//! - the same with a raw-content dictionary cut from the input;
//! - a round trip: libzstd compresses the input (level, checksum and window
//!   from the first byte) and ours must give the input back exactly.

use libfuzzer_sys::fuzz_target;
use urna_format::encoding::{zstd_decompress, zstd_decompress_with_dict};
use zstd::zstd_safe::CParameter;

const LIMIT: usize = 1 << 22;

fn libzstd(src: &[u8], dict: Option<&[u8]>) -> Option<Vec<u8>> {
    let mut d = match dict {
        Some(dict) => zstd::bulk::Decompressor::with_dictionary(dict).ok()?,
        None => zstd::bulk::Decompressor::new().ok()?,
    };
    d.decompress(src, LIMIT).ok()
}

fuzz_target!(|data: &[u8]| {
    let Some((&sel, rest)) = data.split_first() else {
        return;
    };
    match sel % 3 {
        0 => {
            let ours = zstd_decompress(rest, LIMIT);
            if let (Ok(ours), Some(reference)) = (ours, libzstd(rest, None)) {
                assert_eq!(ours, reference, "decoders disagree");
            }
        }
        1 => {
            let cut = (usize::from(sel >> 2) * 16).min(rest.len());
            let (dict, src) = rest.split_at(cut);
            let ours = zstd_decompress_with_dict(src, dict, LIMIT);
            if let (Ok(ours), Some(reference)) = (ours, libzstd(src, Some(dict))) {
                assert_eq!(ours, reference, "decoders disagree with a dictionary");
            }
        }
        _ => {
            let level = i32::from(sel >> 3) % 19 + 1;
            let Ok(mut c) = zstd::bulk::Compressor::new(level) else {
                return;
            };
            let _ = c.set_parameter(CParameter::ChecksumFlag(sel & 4 != 0));
            let Ok(frame) = c.compress(rest) else {
                return;
            };
            match zstd_decompress(&frame, LIMIT) {
                Ok(ours) => assert_eq!(ours, rest, "round trip changed the bytes"),
                Err(e) => panic!("our decoder refused a libzstd frame: {e}"),
            }
        }
    }
});
