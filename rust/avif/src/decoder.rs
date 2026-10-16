//! One AV1 temporal unit in, one picture out, through `urna-av1`'s
//! dav1d-shaped API. The picture's planes are copied out into `Frame`, so
//! the decoder can be reused for the alpha item and dropped freely.

use std::ptr::NonNull;

use urna_av1::include::dav1d::data::Dav1dData;
use urna_av1::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use urna_av1::include::dav1d::headers::{
    DAV1D_MC_BT470BG, DAV1D_MC_BT601, DAV1D_MC_BT709, DAV1D_MC_BT2020_NCL, DAV1D_MC_FCC,
    DAV1D_MC_IDENTITY, DAV1D_MC_SMPTE_YCGCO, DAV1D_MC_SMPTE240, DAV1D_MC_UNKNOWN,
    DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422,
    DAV1D_PIXEL_LAYOUT_I444,
};
use urna_av1::include::dav1d::picture::Dav1dPicture;
use urna_av1::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

use crate::{Error, Result};

/// How the chroma planes are sampled against luma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    Mono,
    C420,
    C422,
    C444,
}

/// The matrix the planes are in, from the sequence header (ITU-T H.273).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Identity,
    Bt709,
    Fcc,
    Bt601,
    Smpte240,
    YCgCo,
    Bt2020Ncl,
}

/// A decoded picture, planes copied out, 8 or 16 bits per sample.
#[derive(Clone, Debug)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub bpc: u8,
    pub chroma: Chroma,
    pub matrix: Option<Matrix>,
    pub full_range: bool,
    /// `[y, u, v]`, each `stride * rows` samples; `u` and `v` empty for mono.
    pub planes: [Vec<u16>; 3],
    pub strides: [usize; 3],
}

pub struct Decoder {
    ctx: Option<Dav1dContext>,
}

impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder").finish_non_exhaustive()
    }
}

fn ok(res: urna_av1::Dav1dResult) -> Result<()> {
    if res.0 == 0 {
        Ok(())
    } else {
        Err(Error::Decoder(res.0))
    }
}

impl Decoder {
    pub fn open() -> Result<Self> {
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        // SAFETY: `dav1d_default_settings` fills the whole struct it is
        // pointed at, and the pointer comes from a live local.
        unsafe { dav1d_default_settings(NonNull::from(&mut settings).cast()) };
        // SAFETY: just initialised by `dav1d_default_settings`.
        let settings = unsafe { settings.assume_init_mut() };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        settings.n_threads = threads.min(8) as _;
        settings.max_frame_delay = 1;
        settings.apply_grain = 1;
        let mut ctx = None;
        // SAFETY: both pointers come from live locals, and `ctx` receives a
        // context this struct closes in `Drop`.
        let res = unsafe {
            dav1d_open(
                Some(NonNull::from(&mut ctx)),
                Some(NonNull::from(&mut *settings)),
            )
        };
        ok(res)?;
        Ok(Self { ctx })
    }

    /// Decodes one AVIF item (one temporal unit); for a layered item the
    /// last picture is the complete image.
    pub fn decode(&mut self, av1: &[u8]) -> Result<Frame> {
        let ctx = self.ctx.ok_or(Error::Unsupported("decoder closed"))?;
        if av1.is_empty() {
            return Err(Error::Unsupported("empty av1 item"));
        }
        let mut data = Dav1dData::default();
        // SAFETY: `data` is a live local the decoder fills; the buffer it
        // returns is `av1.len()` bytes long and copied into once.
        let buf = unsafe { dav1d_data_create(Some(NonNull::from(&mut data)), av1.len()) };
        let Some(buf) = NonNull::new(buf) else {
            return Err(Error::Unsupported("out of memory"));
        };
        // SAFETY: `buf` has room for `av1.len()` bytes, `av1` is that long,
        // and the two never overlap.
        unsafe { std::ptr::copy_nonoverlapping(av1.as_ptr(), buf.as_ptr(), av1.len()) };
        let mut last: Option<Dav1dPicture> = None;
        let sent = loop {
            // SAFETY: `ctx` is open and `data` is the buffer created above.
            let res = unsafe { dav1d_send_data(Some(ctx), Some(NonNull::from(&mut data))) };
            if res.0 == 0 {
                break Ok(());
            }
            let mut pending = Dav1dPicture::default();
            // SAFETY: `pending` is a live default picture the decoder fills
            // or leaves empty; it is unreferenced on both paths.
            let drained =
                unsafe { dav1d_get_picture(Some(ctx), Some(NonNull::from(&mut pending))) };
            if drained.0 != 0 {
                // SAFETY: `pending` was never filled; unref is a no-op on it.
                unsafe { dav1d_picture_unref(Some(NonNull::from(&mut pending))) };
                break ok(res);
            }
            replace(&mut last, pending);
        };
        // SAFETY: `data` is the buffer created above; the decoder holds its
        // own reference to the bytes it took.
        unsafe { dav1d_data_unref(Some(NonNull::from(&mut data))) };
        if let Err(e) = sent {
            unref(last);
            return Err(e);
        }
        loop {
            let mut pic = Dav1dPicture::default();
            // SAFETY: `pic` is a live default picture, filled or left empty.
            let res = unsafe { dav1d_get_picture(Some(ctx), Some(NonNull::from(&mut pic))) };
            if res.0 != 0 {
                // SAFETY: `pic` was never filled.
                unsafe { dav1d_picture_unref(Some(NonNull::from(&mut pic))) };
                let Some(mut got) = last else {
                    ok(res)?;
                    return Err(Error::Unsupported("no frame decoded"));
                };
                let frame = copy_out(&got);
                // SAFETY: `got` was filled by `dav1d_get_picture` and is not
                // used after this.
                unsafe { dav1d_picture_unref(Some(NonNull::from(&mut got))) };
                return frame;
            }
            replace(&mut last, pic);
        }
    }
}

fn replace(last: &mut Option<Dav1dPicture>, pic: Dav1dPicture) {
    unref(last.replace(pic));
}

fn unref(pic: Option<Dav1dPicture>) {
    if let Some(mut p) = pic {
        // SAFETY: `p` was filled by `dav1d_get_picture` and is dropped here.
        unsafe { dav1d_picture_unref(Some(NonNull::from(&mut p))) };
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: closes the context opened in `open`, once.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.ctx))) };
    }
}

/// Copies the picture's planes into owned `u16` samples.
fn copy_out(pic: &Dav1dPicture) -> Result<Frame> {
    let width = usize::try_from(pic.p.w).map_err(|_| Error::Unsupported("picture width"))?;
    let height = usize::try_from(pic.p.h).map_err(|_| Error::Unsupported("picture height"))?;
    if width == 0 || height == 0 {
        return Err(Error::Unsupported("empty picture"));
    }
    let bpc = u8::try_from(pic.p.bpc).map_err(|_| Error::Unsupported("bit depth"))?;
    let bytes = if bpc <= 8 { 1 } else { 2 };
    let (chroma, ss_x, ss_y) = match pic.p.layout {
        l if l == DAV1D_PIXEL_LAYOUT_I400 => (Chroma::Mono, 0, 0),
        l if l == DAV1D_PIXEL_LAYOUT_I420 => (Chroma::C420, 1, 1),
        l if l == DAV1D_PIXEL_LAYOUT_I422 => (Chroma::C422, 1, 0),
        l if l == DAV1D_PIXEL_LAYOUT_I444 => (Chroma::C444, 0, 0),
        _ => return Err(Error::Unsupported("pixel layout")),
    };
    let (matrix, full_range) = match pic.seq_hdr {
        Some(h) => {
            // SAFETY: the decoder keeps the sequence header alive as long
            // as the picture that points at it, and `pic` is alive here.
            let h = unsafe { h.as_ref() };
            let m = match h.mtrx {
                m if m == DAV1D_MC_IDENTITY => Some(Matrix::Identity),
                m if m == DAV1D_MC_BT709 => Some(Matrix::Bt709),
                m if m == DAV1D_MC_FCC => Some(Matrix::Fcc),
                m if m == DAV1D_MC_BT470BG || m == DAV1D_MC_BT601 => Some(Matrix::Bt601),
                m if m == DAV1D_MC_SMPTE240 => Some(Matrix::Smpte240),
                m if m == DAV1D_MC_SMPTE_YCGCO => Some(Matrix::YCgCo),
                m if m == DAV1D_MC_BT2020_NCL => Some(Matrix::Bt2020Ncl),
                m if m == DAV1D_MC_UNKNOWN => None,
                _ => return Err(Error::Unsupported("matrix coefficients")),
            };
            (m, h.color_range != 0)
        }
        None => (None, false),
    };
    let mut planes: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut strides = [0usize; 3];
    let n_planes = if chroma == Chroma::Mono { 1 } else { 3 };
    for (i, out) in planes.iter_mut().enumerate().take(n_planes) {
        let (w, h) = if i == 0 {
            (width, height)
        } else {
            ((width + ss_x) >> ss_x, (height + ss_y) >> ss_y)
        };
        let stride_bytes = pic.stride[usize::from(i != 0)];
        let stride = usize::try_from(stride_bytes)
            .ok()
            .filter(|s| s % bytes == 0 && *s / bytes >= w)
            .ok_or(Error::Unsupported("plane stride"))?
            / bytes;
        let Some(ptr) = pic.data[i] else {
            return Err(Error::Unsupported("missing plane"));
        };
        let mut v = Vec::with_capacity(w * h);
        for row in 0..h {
            // SAFETY: rav1d allocates whole rows of `stride` samples for
            // `h` rows, so `row * stride .. row * stride + w` is inside the
            // plane; the picture is alive for the whole copy.
            let samples = unsafe {
                let base = ptr.as_ptr().cast::<u8>().add(row * stride * bytes);
                std::slice::from_raw_parts(base, w * bytes)
            };
            if bytes == 1 {
                v.extend(samples.iter().map(|&s| u16::from(s)));
            } else {
                v.extend(
                    samples
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| u16::from_ne_bytes(*c)),
                );
            }
        }
        *out = v;
        strides[i] = w;
    }
    Ok(Frame {
        width,
        height,
        bpc,
        chroma,
        matrix,
        full_range,
        planes,
        strides,
    })
}
