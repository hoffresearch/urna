//! An AVIF still image to RGBA pixels: the ISOBMFF container is read here
//! (`boxes`), the AV1 payload is decoded by `urna-av1`, and the planes are
//! converted here (`color`). What a `.urna` file stores as an inlined blob
//! is one primary item, optionally one alpha item; that is what this
//! decodes, with every length checked against the bytes given before any
//! allocation.

mod boxes;
mod color;
pub mod decoder;

pub use boxes::{Container, Properties, parse};
pub use decoder::{Chroma, Decoder, Frame, Matrix};

/// Decoded pixels, 8 bits per channel, rows top to bottom with no padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("avif: truncated at byte {at}: {what}")]
    Truncated { at: usize, what: &'static str },
    #[error("avif: box {name} declares {declared} bytes, {available} available")]
    HostileSize {
        name: String,
        declared: u64,
        available: usize,
    },
    #[error("avif: not an avif file ({0})")]
    NotAvif(&'static str),
    #[error("avif: {0} box missing")]
    Missing(&'static str),
    #[error("avif: {0}")]
    Malformed(&'static str),
    #[error("av1: decoder error {0}")]
    Decoder(i32),
    #[error("av1: {0}")]
    Unsupported(&'static str),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Decodes a whole AVIF file held in memory.
pub fn decode(bytes: &[u8]) -> Result<Image> {
    let container = parse(bytes)?;
    let mut dec = decoder::Decoder::open()?;
    let frame = dec.decode(container.primary)?;
    let mut image = color::to_rgba(&frame)?;
    if let Some(alpha) = container.alpha {
        let a = dec.decode(alpha)?;
        color::write_alpha(&mut image, &a, container.premultiplied_alpha)?;
    }
    Ok(image)
}
