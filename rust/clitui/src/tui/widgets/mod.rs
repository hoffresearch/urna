//! The ui's own widgets over ratatui: a text input with a cursor, a
//! spinner, the key-help footer, a scrollbar, an OSC 8 link and the file
//! picker. Each is a few dozen lines drawn straight into the buffer, the
//! whole of what the screens need and nothing a dependency would add.

mod help;
mod input;
mod link;
mod picker;
mod scrollbar;
mod spinner;

pub use help::{Binding, Help, HelpStyles};
pub use input::{Input, InputState, InputStyles};
pub use link::link;
pub use picker::Picker;
pub use scrollbar::ScrollBar;
pub use spinner::Spinner;

#[cfg(test)]
mod tests {
    //! every widget drawn into a tiny and a large buffer, without a panic.
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Style;
    use ratatui::widgets::StatefulWidget;
    use std::time::Duration;

    fn frame(w: u16, h: u16) {
        let area = Rect::new(0, 0, w, h);
        let mut buf = Buffer::empty(area);
        let mut state = InputState::new();
        state.set_value("a query that is longer than a small frame".into());
        state.set_focused(true);
        let input = Input::new("ask")
            .description("desc")
            .placeholder("type")
            .prompt("› ");
        StatefulWidget::render(&input, area, &mut buf, &mut state);
        let mut spin = Spinner::dots();
        spin.tick(Duration::from_millis(500));
        assert!(!spin.frame_str().is_empty());
        let help = Help::new(vec![
            Binding::new("enter", "ask"),
            Binding::new("↑↓", "hit"),
            Binding::new("q", "quit"),
        ]);
        help.render(Rect::new(0, 0, w, 1), &mut buf);
        ScrollBar::vertical(120, usize::from(h))
            .offset(40)
            .track_style(Style::default())
            .thumb_style(Style::default())
            .render(Rect::new(w.saturating_sub(1), 0, 1, h), &mut buf);
        link(
            &mut buf,
            0,
            h.saturating_sub(1),
            "docs",
            "https://example.invalid",
            w,
        );
        let dir = std::env::temp_dir();
        let picker = Picker::new(&dir, |e| e.is_dir).expect("lists the temp dir");
        picker.render(&mut buf, area, " open ");
    }

    #[test]
    fn every_widget_draws_small_and_large_without_panicking() {
        frame(20, 5);
        frame(200, 60);
        frame(1, 1);
    }
}
