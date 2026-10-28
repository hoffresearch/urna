//! Planes to RGBA, 8 bits per channel. The matrix (ITU-T H.273) and the
//! range come from the sequence header; chroma is upsampled by nearest
//! sample, as the fast paths of libyuv and dav1d's own tools do. The
//! arithmetic is fixed point, 14 fractional bits, rounded to nearest.

use crate::decoder::{Chroma, Frame, Matrix};
use crate::{Error, Image, Result};

const FRAC: i32 = 14;
const HALF: i32 = 1 << (FRAC - 1);

/// Kr and Kb of each matrix, from H.273 table 4.
fn kr_kb(m: Matrix) -> Option<(f64, f64)> {
    Some(match m {
        Matrix::Bt709 => (0.2126, 0.0722),
        Matrix::Fcc => (0.30, 0.11),
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Smpte240 => (0.212, 0.087),
        Matrix::Bt2020Ncl => (0.2627, 0.0593),
        Matrix::Identity | Matrix::YCgCo => return None,
    })
}

/// The fixed-point conversion: `r = y_scale * y' + cr_r * cr'` and so on,
/// with `y'` and the chroma already offset.
#[derive(Clone, Copy, Debug)]
struct Coef {
    y_scale: i32,
    c_scale_num: f64,
    cr_r: i32,
    cb_g: i32,
    cr_g: i32,
    cb_b: i32,
    y_offset: i32,
    max: i32,
}

fn coefficients(m: Matrix, full_range: bool, bits: u8) -> Option<Coef> {
    let (kr, kb) = kr_kb(m)?;
    let kg = 1.0 - kr - kb;
    let max = (1i32 << bits) - 1;
    let (y_offset, y_span, c_span) = if full_range {
        (0, f64::from(max), f64::from(max))
    } else {
        (
            16 << (bits - 8),
            f64::from(219 << (bits - 8)),
            f64::from(224 << (bits - 8)),
        )
    };
    let scale = |v: f64| (v * f64::from(1 << FRAC)).round() as i32;
    let y_scale = f64::from(max) / y_span;
    let c_scale = f64::from(max) / c_span;
    Some(Coef {
        y_scale: scale(y_scale),
        c_scale_num: c_scale,
        cr_r: scale(2.0 * (1.0 - kr) * c_scale),
        cb_g: scale(2.0 * kb * (1.0 - kb) / kg * c_scale),
        cr_g: scale(2.0 * kr * (1.0 - kr) / kg * c_scale),
        cb_b: scale(2.0 * (1.0 - kb) * c_scale),
        y_offset,
        max,
    })
}

/// Reads sample `(x, y)` of plane `p`, subsampled as `chroma` says.
fn sample(frame: &Frame, p: usize, x: usize, y: usize) -> i32 {
    let (sx, sy) = match (p, frame.chroma) {
        (0, _) | (_, Chroma::C444 | Chroma::Mono) => (x, y),
        (_, Chroma::C420) => (x >> 1, y >> 1),
        (_, Chroma::C422) => (x >> 1, y),
    };
    i32::from(frame.planes[p][sy * frame.strides[p] + sx])
}

fn to8(v: i32, max: i32) -> u8 {
    let v = v.clamp(0, max);
    if max == 255 {
        v as u8
    } else {
        ((v * 255 + max / 2) / max) as u8
    }
}

pub fn to_rgba(frame: &Frame) -> Result<Image> {
    let bits = frame.bpc;
    if !(8..=16).contains(&bits) {
        return Err(Error::Unsupported("bit depth"));
    }
    let max = (1i32 << bits) - 1;
    let (w, h) = (frame.width, frame.height);
    let mut rgba = Vec::with_capacity(w * h * 4);
    match frame.chroma {
        Chroma::Mono => {
            let (lo, span) = if frame.full_range {
                (0, max)
            } else {
                (16 << (bits - 8), 219 << (bits - 8))
            };
            for y in 0..h {
                for x in 0..w {
                    let v = (sample(frame, 0, x, y) - lo).clamp(0, span);
                    let g = to8((v * max + span / 2) / span, max);
                    rgba.extend_from_slice(&[g, g, g, 255]);
                }
            }
        }
        _ => {
            let matrix = frame.matrix.unwrap_or(Matrix::Bt601);
            match matrix {
                Matrix::Identity => {
                    if frame.chroma != Chroma::C444 {
                        return Err(Error::Unsupported("identity matrix with subsampled chroma"));
                    }
                    for y in 0..h {
                        for x in 0..w {
                            let g = to8(sample(frame, 0, x, y), max);
                            let b = to8(sample(frame, 1, x, y), max);
                            let r = to8(sample(frame, 2, x, y), max);
                            rgba.extend_from_slice(&[r, g, b, 255]);
                        }
                    }
                }
                Matrix::YCgCo => {
                    let half = 1 << (bits - 1);
                    for y in 0..h {
                        for x in 0..w {
                            let yy = sample(frame, 0, x, y);
                            let cg = sample(frame, 1, x, y) - half;
                            let co = sample(frame, 2, x, y) - half;
                            let t = yy - cg;
                            rgba.extend_from_slice(&[
                                to8(t + co, max),
                                to8(yy + cg, max),
                                to8(t - co, max),
                                255,
                            ]);
                        }
                    }
                }
                m => {
                    let c = coefficients(m, frame.full_range, bits)
                        .ok_or(Error::Unsupported("matrix coefficients"))?;
                    let _ = c.c_scale_num;
                    let half = 1 << (bits - 1);
                    for y in 0..h {
                        for x in 0..w {
                            let yy = (sample(frame, 0, x, y) - c.y_offset) * c.y_scale;
                            let cb = sample(frame, 1, x, y) - half;
                            let cr = sample(frame, 2, x, y) - half;
                            let r = (yy + c.cr_r * cr + HALF) >> FRAC;
                            let g = (yy - c.cb_g * cb - c.cr_g * cr + HALF) >> FRAC;
                            let b = (yy + c.cb_b * cb + HALF) >> FRAC;
                            rgba.extend_from_slice(&[
                                to8(r, c.max),
                                to8(g, c.max),
                                to8(b, c.max),
                                255,
                            ]);
                        }
                    }
                }
            }
        }
    }
    Ok(Image {
        width: w as u32,
        height: h as u32,
        rgba,
    })
}

/// The alpha item's luma becomes the alpha channel; premultiplied color is
/// divided back out.
pub fn write_alpha(image: &mut Image, alpha: &Frame, premultiplied: bool) -> Result<()> {
    if alpha.width != image.width as usize || alpha.height != image.height as usize {
        return Err(Error::Unsupported("alpha size differs from the image"));
    }
    let bits = alpha.bpc;
    let max = (1i32 << bits) - 1;
    let (lo, span) = if alpha.full_range {
        (0, max)
    } else {
        (16 << (bits - 8), 219 << (bits - 8))
    };
    for y in 0..alpha.height {
        for x in 0..alpha.width {
            let v = (sample(alpha, 0, x, y) - lo).clamp(0, span);
            let a = to8((v * max + span / 2) / span, max);
            let px = &mut image.rgba[(y * alpha.width + x) * 4..][..4];
            px[3] = a;
            if premultiplied && a != 0 && a != 255 {
                for c in &mut px[..3] {
                    *c = ((u16::from(*c) * 255 / u16::from(a)).min(255)) as u8;
                }
            }
        }
    }
    Ok(())
}
