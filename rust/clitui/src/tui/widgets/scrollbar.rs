//! A vertical scrollbar: a track, and a thumb whose size and place follow
//! the viewport over the content.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

#[derive(Clone, Debug, Default)]
pub struct ScrollBar {
    content_len: usize,
    viewport_len: usize,
    offset: usize,
    track: Style,
    thumb: Style,
}

impl ScrollBar {
    pub fn vertical(content_len: usize, viewport_len: usize) -> Self {
        Self {
            content_len,
            viewport_len,
            ..Default::default()
        }
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn track_style(mut self, s: Style) -> Self {
        self.track = s;
        self
    }

    pub fn thumb_style(mut self, s: Style) -> Self {
        self.thumb = s;
        self
    }

    /// Draw in the first column of `area`; nothing when the content fits.
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let h = usize::from(area.height);
        if h == 0 || area.width == 0 || self.content_len <= self.viewport_len {
            return;
        }
        let thumb_len = (h * self.viewport_len / self.content_len).clamp(1, h);
        let travel = h - thumb_len;
        let max_offset = self.content_len - self.viewport_len;
        let thumb_at = if max_offset == 0 {
            0
        } else {
            (travel * self.offset.min(max_offset)).div_ceil(max_offset)
        };
        for row in 0..h {
            let y = area.y + row as u16;
            let on = (thumb_at..thumb_at + thumb_len).contains(&row);
            if let Some(cell) = buf.cell_mut((area.x, y)) {
                cell.set_symbol(if on { "┃" } else { "│" });
                cell.set_style(if on { self.thumb } else { self.track });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_thumb_reaches_both_ends_and_a_fitting_list_draws_nothing() {
        let area = Rect::new(0, 0, 1, 10);
        let col = |offset: usize| {
            let mut buf = Buffer::empty(area);
            ScrollBar::vertical(100, 10)
                .offset(offset)
                .render(area, &mut buf);
            (0..10)
                .map(|y| buf[(0, y)].symbol().to_string())
                .collect::<String>()
        };
        assert_eq!(col(0), "┃│││││││││");
        assert_eq!(col(90), "│││││││││┃");
        assert_eq!(col(45), "│││││┃││││");
        let mut buf = Buffer::empty(area);
        ScrollBar::vertical(5, 10).render(area, &mut buf);
        assert_eq!(buf[(0, 0)].symbol(), " ");
    }
}
