//! An AVIF still image to RGBA pixels: the ISOBMFF container is read here
//! (`boxes`), the AV1 payload is decoded by `urna-av1dec`, and the planes are
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

#[derive(Debug)]
pub enum Error {
    Truncated {
        at: usize,
        what: &'static str,
    },
    HostileSize {
        name: String,
        declared: u64,
        available: usize,
    },
    NotAvif(&'static str),
    Missing(&'static str),
    Malformed(&'static str),
    Decoder(i32),
    Unsupported(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { at, what } => write!(f, "avif: truncated at byte {at}: {what}"),
            Self::HostileSize {
                name,
                declared,
                available,
            } => write!(
                f,
                "avif: box {name} declares {declared} bytes, {available} available"
            ),
            Self::NotAvif(v0) => write!(f, "avif: not an avif file ({v0})"),
            Self::Missing(v0) => write!(f, "avif: {v0} box missing"),
            Self::Malformed(v0) => write!(f, "avif: {v0}"),
            Self::Decoder(v0) => write!(f, "av1: decoder error {v0}"),
            Self::Unsupported(v0) => write!(f, "av1: {v0}"),
        }
    }
}

impl std::error::Error for Error {}

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
