//! A one-line text input: a title with its description above, a prompt,
//! the text with a block cursor, or a placeholder while empty. The state
//! keeps the text and a cursor in characters; the widget only draws.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::StatefulWidget;

use crate::tui::hud;

#[derive(Clone, Debug, Default)]
pub struct InputState {
    value: String,
    /// the cursor, in characters from the start.
    cursor: usize,
    focused: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn set_value(&mut self, value: String) {
        self.cursor = value.chars().count();
        self.value = value;
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    fn byte_at(&self, chars: usize) -> usize {
        self.value
            .char_indices()
            .nth(chars)
            .map_or(self.value.len(), |(i, _)| i)
    }

    pub fn insert_char(&mut self, c: char) {
        let at = self.byte_at(self.cursor);
        self.value.insert(at, c);
        self.cursor += 1;
    }

    /// Backspace: the character before the cursor.
    pub fn delete_before(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let start = self.byte_at(self.cursor - 1);
        let end = self.byte_at(self.cursor);
        self.value.replace_range(start..end, "");
        self.cursor -= 1;
    }

    /// Delete: the character under the cursor.
    pub fn delete_at(&mut self) {
        let start = self.byte_at(self.cursor);
        if start >= self.value.len() {
            return;
        }
        let end = self.byte_at(self.cursor + 1);
        self.value.replace_range(start..end, "");
    }

    pub fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.value.chars().count());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.value.chars().count();
    }
}

#[derive(Clone, Debug, Default)]
pub struct InputStyles {
    pub title: Style,
    pub description: Style,
    pub prompt: Style,
    pub text: Style,
    pub placeholder: Style,
    pub cursor: Style,
}

#[derive(Clone, Debug, Default)]
pub struct Input<'a> {
    title: &'a str,
    description: &'a str,
    placeholder: &'a str,
    prompt: &'a str,
    styles: InputStyles,
}

impl<'a> Input<'a> {
    pub fn new(title: &'a str) -> Self {
        Self {
            title,
            ..Default::default()
        }
    }

    pub fn description(mut self, d: &'a str) -> Self {
        self.description = d;
        self
    }

    pub fn placeholder(mut self, p: &'a str) -> Self {
        self.placeholder = p;
        self
    }

    pub fn prompt(mut self, p: &'a str) -> Self {
        self.prompt = p;
        self
    }

    pub fn styles(mut self, s: InputStyles) -> Self {
        self.styles = s;
        self
    }
}

impl StatefulWidget for &Input<'_> {
    type State = InputState;

    /// Row 0: the title and its description. Row 1: the prompt and the
    /// text, scrolled so the cursor is visible, the cursor cell in its
    /// style; the placeholder while the text is empty.
    fn render(self, area: Rect, buf: &mut Buffer, state: &mut InputState) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let s = &self.styles;
        let x = hud::put(buf, area.x, area.y, self.title, s.title, area.width);
        if !self.description.is_empty() && x + 2 < area.right() {
            hud::put(
                buf,
                x + 2,
                area.y,
                self.description,
                s.description,
                area.right() - x - 2,
            );
        }
        if area.height < 2 {
            return;
        }
        let y = area.y + 1;
        let px = hud::put(buf, area.x, y, self.prompt, s.prompt, area.width);
        let room = usize::from(area.right().saturating_sub(px));
        if room == 0 {
            return;
        }
        if state.value.is_empty() {
            hud::put(buf, px, y, self.placeholder, s.placeholder, room as u16);
            if state.focused
                && let Some(cell) = buf.cell_mut((px, y))
            {
                cell.set_style(s.cursor);
            }
            return;
        }
        let chars: Vec<char> = state.value.chars().collect();
        // the window of the text that keeps the cursor in view, with one
        // cell spare for a cursor past the end.
        let visible = room.saturating_sub(1).max(1);
        let start = state.cursor.saturating_sub(visible);
        let shown: String = chars.iter().skip(start).take(visible).collect();
        hud::put(buf, px, y, &shown, s.text, room as u16);
        if state.focused {
            let cx = px + (state.cursor - start) as u16;
            if cx < area.right()
                && let Some(cell) = buf.cell_mut((cx, y))
            {
                cell.set_style(s.cursor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_move_the_cursor_by_characters_not_bytes() {
        let mut s = InputState::new();
        for c in "raio é".chars() {
            s.insert_char(c);
        }
        assert_eq!(s.value(), "raio é");
        s.move_left();
        s.insert_char('x');
        assert_eq!(s.value(), "raio xé");
        s.delete_before();
        s.delete_at();
        assert_eq!(s.value(), "raio ");
        s.home();
        s.delete_at();
        assert_eq!(s.value(), "aio ");
        s.end();
        s.move_right();
        s.delete_before();
        assert_eq!(s.value(), "aio");
        s.set_value("relâmpago".into());
        s.delete_before();
        assert_eq!(s.value(), "relâmpag");
    }
}
