//! Sixel (xterm, foot, mlterm, wezterm, iTerm2): a DCS `ESC P q` stream
//! of a palette of up to 256 colors, then bands of six rows where each
//! color's pixels are written as characters `?`..`~` (bit n of the six
//! vertical pixels), with `!n` run-length repeats, `$` back to the band's
//! start for the next color and `-` down to the next band. The palette
//! is a median cut over the image's colors: the box with the widest
//! channel splits at its median by pixel count until 256 boxes remain.

use super::Rgba;

const COLORS: usize = 256;

/// The palette (RGB) and one index per pixel.
fn quantize(img: &Rgba) -> (Vec<[u8; 3]>, Vec<u8>) {
    // count colors at 5 bits per channel: 32768 bins at most, enough to
    // build the boxes; the exact colors come back through the box means.
    let mut counts: std::collections::HashMap<u16, (u64, [u64; 3])> =
        std::collections::HashMap::new();
    for p in img.data.as_chunks::<4>().0 {
        let key = (u16::from(p[0] >> 3) << 10) | (u16::from(p[1] >> 3) << 5) | u16::from(p[2] >> 3);
        let e = counts.entry(key).or_insert((0, [0; 3]));
        e.0 += 1;
        for (sum, v) in e.1.iter_mut().zip(p) {
            *sum += u64::from(*v);
        }
    }
    // a color: its mean channels and its weight.
    let mut colors: Vec<([u8; 3], u64)> = counts
        .values()
        .map(|(n, sum)| {
            let m = |c: usize| (sum[c] / *n) as u8;
            ([m(0), m(1), m(2)], *n)
        })
        .collect();
    colors.sort();
    let mut boxes: Vec<Vec<([u8; 3], u64)>> = vec![colors];
    while boxes.len() < COLORS {
        // split the box with the widest channel range.
        let (bi, ch) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let (w, c) = (0..3)
                    .map(|c| {
                        let (lo, hi) = b.iter().fold((255u8, 0u8), |(lo, hi), (p, _)| {
                            (lo.min(p[c]), hi.max(p[c]))
                        });
                        (hi - lo, c)
                    })
                    .max()
                    .unwrap_or((0, 0));
                (w, i, c)
            })
            .max()
            .map(|(_, i, c)| (i, c))
            .unwrap_or((usize::MAX, 0));
        if bi == usize::MAX {
            break;
        }
        let mut b = boxes.swap_remove(bi);
        b.sort_by_key(|(p, _)| p[ch]);
        let total: u64 = b.iter().map(|(_, n)| *n).sum();
        let mut acc = 0u64;
        let mut cut = 1;
        for (i, (_, n)) in b.iter().enumerate() {
            acc += n;
            if acc * 2 >= total {
                cut = (i + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let rest = b.split_off(cut);
        boxes.push(b);
        boxes.push(rest);
    }
    let palette: Vec<[u8; 3]> = boxes
        .iter()
        .map(|b| {
            let total: u64 = b.iter().map(|(_, n)| *n).sum::<u64>().max(1);
            let mean =
                |c: usize| (b.iter().map(|(p, n)| u64::from(p[c]) * n).sum::<u64>() / total) as u8;
            [mean(0), mean(1), mean(2)]
        })
        .collect();
    let nearest = |p: &[u8]| -> u8 {
        let mut best = (u32::MAX, 0usize);
        for (i, c) in palette.iter().enumerate() {
            let d = (0..3)
                .map(|k| {
                    let d = i32::from(p[k]) - i32::from(c[k]);
                    (d * d) as u32
                })
                .sum::<u32>();
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1 as u8
    };
    let indexes = img
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| nearest(p))
        .collect();
    (palette, indexes)
}

pub fn encode(img: &Rgba) -> Vec<u8> {
    let (palette, idx) = quantize(img);
    let (w, h) = (img.width as usize, img.height as usize);
    let mut out = format!("\x1bPq\"1;1;{w};{h}").into_bytes();
    for (i, c) in palette.iter().enumerate() {
        let pct = |v: u8| (u32::from(v) * 100 + 127) / 255;
        out.extend_from_slice(
            format!("#{i};2;{};{};{}", pct(c[0]), pct(c[1]), pct(c[2])).as_bytes(),
        );
    }
    let mut band = vec![0u8; w];
    for y0 in (0..h).step_by(6) {
        let rows = (h - y0).min(6);
        let mut used = vec![false; palette.len()];
        for y in y0..y0 + rows {
            for x in 0..w {
                used[usize::from(idx[y * w + x])] = true;
            }
        }
        let mut first = true;
        for (color, _) in used.iter().enumerate().filter(|(_, u)| **u) {
            if !first {
                out.push(b'$');
            }
            first = false;
            for (x, cell) in band.iter_mut().enumerate() {
                let mut bits = 0u8;
                for (r, y) in (y0..y0 + rows).enumerate() {
                    if usize::from(idx[y * w + x]) == color {
                        bits |= 1 << r;
                    }
                }
                *cell = bits;
            }
            out.extend_from_slice(format!("#{color}").as_bytes());
            let mut x = 0;
            while x < w {
                let v = band[x];
                let mut run = 1;
                while x + run < w && band[x + run] == v {
                    run += 1;
                }
                let ch = b'?' + v;
                if run > 3 {
                    out.extend_from_slice(format!("!{run}").as_bytes());
                    out.push(ch);
                } else {
                    out.extend(std::iter::repeat_n(ch, run));
                }
                x += run;
            }
        }
        out.push(b'-');
    }
    out.extend_from_slice(b"\x1b\\");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_by_two_image_names_its_colors_and_its_one_band() {
        // red, green / blue, red: three palette entries, one band.
        let img = Rgba::new(
            2,
            2,
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255,
            ],
        )
        .expect("2x2");
        let text = String::from_utf8(encode(&img)).expect("ascii");
        assert!(text.starts_with("\x1bPq\"1;1;2;2#"), "{text}");
        assert!(text.ends_with("-\x1b\\"), "{text}");
        assert_eq!(quantize(&img).0.len(), 3, "{text}");
        // red is the top-left and bottom-right pixel: bits 0 and 1 across
        // the two columns, so its run is `@` (bit 0) then `A` (bit 1).
        let (palette, idx) = quantize(&img);
        let red = palette.iter().position(|c| c[0] > 200).expect("red");
        assert_eq!(idx, vec![red as u8, idx[1], idx[2], red as u8]);
        assert!(text.contains(&format!("#{red}@A")), "{text}");
        assert_eq!(text.matches('$').count(), 2);
    }

    #[test]
    fn many_colors_fold_into_at_most_256() {
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
        let (palette, idx) = quantize(&img);
        assert!(palette.len() <= 256 && palette.len() > 100);
        assert!(idx.iter().all(|&i| usize::from(i) < palette.len()));
    }
}
