//! iTerm2's inline images (iTerm2, WezTerm, mintty): one OSC 1337
//! `File=` command with the image as a base64 PNG, drawn at the cursor
//! and scaled into `width` x `height` cells with the aspect kept.

use super::{Rgba, base64, png};

pub fn encode(img: &Rgba, cols: u16, rows: u16) -> Vec<u8> {
    let payload = base64::encode(&png::encode(img));
    let mut out = format!(
        "\x1b]1337;File=inline=1;width={cols};height={rows};preserveAspectRatio=1;size={}:",
        payload.len()
    )
    .into_bytes();
    out.extend_from_slice(&payload);
    out.push(0x07);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_by_two_image_is_one_osc_with_a_png_inside() {
        let img = Rgba::new(
            2,
            2,
            vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 0],
        )
        .expect("2x2");
        let text = String::from_utf8(encode(&img, 4, 2)).expect("ascii");
        assert!(
            text.starts_with(
                "\x1b]1337;File=inline=1;width=4;height=2;preserveAspectRatio=1;size="
            )
        );
        assert!(text.ends_with('\x07'));
        // the png signature in base64 is always iVBORw0KGgo.
        assert!(text.contains(":iVBORw0KGgo"), "{text}");
    }
}
