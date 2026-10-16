//! The AV1 planes this crate's embedded decoder produces against the same
//! decoder from crates.io, sample by sample, on every AVIF blob of the
//! corpus named by `URNA_CARDS_CORPUS` (skipped without it). The decode is
//! defined bit-exactly by the AV1 specification, so any difference is a
//! bug in the fork.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test: a failure is the finding"
)]

use std::ptr::NonNull;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::Dav1dSettings;
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};

/// Decodes one item with the crates.io rav1d and returns its planes as the
/// embedded decoder's `Frame` lays them out: `w` samples per row, no stride.
fn upstream_planes(av1: &[u8]) -> (usize, usize, [Vec<u16>; 3]) {
    // SAFETY: the dav1d C-shaped API, driven the way `urna_avif::decoder`
    // drives the embedded copy: every pointer is a live local, every
    // picture and buffer is unreferenced once, the context closed once.
    unsafe {
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        dav1d_default_settings(NonNull::from(&mut settings).cast());
        let settings = settings.assume_init_mut();
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        settings.apply_grain = 1;
        let mut ctx = None;
        assert_eq!(
            dav1d_open(
                Some(NonNull::from(&mut ctx)),
                Some(NonNull::from(&mut *settings))
            )
            .0,
            0
        );
        let ctx = ctx.expect("open");
        let mut data = Dav1dData::default();
        let buf = dav1d_data_create(Some(NonNull::from(&mut data)), av1.len());
        std::ptr::copy_nonoverlapping(av1.as_ptr(), buf, av1.len());
        assert_eq!(
            dav1d_send_data(Some(ctx), Some(NonNull::from(&mut data))).0,
            0
        );
        dav1d_data_unref(Some(NonNull::from(&mut data)));
        let mut pic = Dav1dPicture::default();
        assert_eq!(
            dav1d_get_picture(Some(ctx), Some(NonNull::from(&mut pic))).0,
            0
        );
        let (w, h) = (pic.p.w as usize, pic.p.h as usize);
        let bytes = if pic.p.bpc <= 8 { 1 } else { 2 };
        let mut planes: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (i, out) in planes.iter_mut().enumerate() {
            let (pw, ph) = if i == 0 {
                (w, h)
            } else {
                (w.div_ceil(2), h.div_ceil(2))
            };
            let stride = pic.stride[usize::from(i != 0)] as usize;
            let ptr = pic.data[i].expect("plane").as_ptr().cast::<u8>();
            for row in 0..ph {
                let r = std::slice::from_raw_parts(ptr.add(row * stride), pw * bytes);
                if bytes == 1 {
                    out.extend(r.iter().map(|&s| u16::from(s)));
                } else {
                    out.extend(r.as_chunks::<2>().0.iter().map(|c| u16::from_ne_bytes(*c)));
                }
            }
        }
        dav1d_picture_unref(Some(NonNull::from(&mut pic)));
        let mut ctx = Some(ctx);
        dav1d_close(Some(NonNull::from(&mut ctx)));
        (w, h, planes)
    }
}

#[test]
fn the_embedded_decoder_produces_the_upstream_planes() {
    let Some(path) = std::env::var_os("URNA_CARDS_CORPUS") else {
        eprintln!("URNA_CARDS_CORPUS unset: plane parity skipped");
        return;
    };
    let limit: usize = std::env::var("URNA_PARITY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let rt = urna_engine::MmapUrnaFile::open(std::path::Path::new(&path)).expect("corpus opens");
    let refs = rt.blob_refs().expect("the corpus has blobs");
    let avif: Vec<usize> = refs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.inlined && r.original_uri.ends_with(".avif"))
        .map(|(i, _)| i)
        .take(limit)
        .collect();
    let mut dec = urna_avif::Decoder::open().expect("decoder");
    let mut equal = 0usize;
    for &i in &avif {
        let bytes = rt.blob_bytes(i).expect("blob bytes");
        let container = urna_avif::parse(bytes).unwrap_or_else(|e| panic!("blob {i}: {e}"));
        let ours = dec
            .decode(container.primary)
            .unwrap_or_else(|e| panic!("blob {i}: {e}"));
        assert_eq!(
            ours.chroma,
            urna_avif::Chroma::C420,
            "blob {i}: the corpus is 4:2:0"
        );
        let (w, h, theirs) = upstream_planes(container.primary);
        assert_eq!((ours.width, ours.height), (w, h), "blob {i}: size");
        for (p, (a, b)) in ours.planes.iter().zip(&theirs).enumerate() {
            assert!(a == b, "blob {i}: plane {p} differs");
        }
        equal += 1;
    }
    eprintln!("plane parity over {} blobs: {equal} identical", avif.len());
}
