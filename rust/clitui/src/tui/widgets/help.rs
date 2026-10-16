//! The key-help footer: `key desc · key desc …`, cut with an ellipsis when
//! the row is too narrow for every binding.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;

use crate::tui::hud;

#[derive(Clone, Debug)]
pub struct Binding {
    pub key: String,
    pub desc: String,
}

impl Binding {
    pub fn new(key: &str, desc: &str) -> Self {
        Self {
            key: key.into(),
            desc: desc.into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HelpStyles {
    pub key: Style,
    pub desc: Style,
    pub separator: Style,
    pub ellipsis: Style,
}

#[derive(Clone, Debug, Default)]
pub struct Help {
    bindings: Vec<Binding>,
    styles: HelpStyles,
    separator: String,
}

impl Help {
    pub fn new(bindings: Vec<Binding>) -> Self {
        Self {
            bindings,
            styles: HelpStyles::default(),
            separator: " · ".into(),
        }
    }

    pub fn styles(mut self, s: HelpStyles) -> Self {
        self.styles = s;
        self
    }

    pub fn separator(mut self, s: &str) -> Self {
        self.separator = s.into();
        self
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let s = &self.styles;
        let right = area.right();
        let mut x = area.x;
        for (i, b) in self.bindings.iter().enumerate() {
            let sep = if i == 0 { "" } else { self.separator.as_str() };
            let need =
                (sep.chars().count() + b.key.chars().count() + 1 + b.desc.chars().count()) as u16;
            let last = i + 1 == self.bindings.len();
            // keep one cell for the ellipsis unless this binding is the last.
            let room = right.saturating_sub(x).saturating_sub(u16::from(!last));
            if need > room {
                if x < right {
                    hud::put(buf, x, area.y, "…", s.ellipsis, right - x);
                }
                return;
            }
            x = hud::put(buf, x, area.y, sep, s.separator, right - x);
            x = hud::put(buf, x, area.y, &b.key, s.key, right - x);
            x = hud::put(buf, x, area.y, " ", s.desc, right - x);
            x = hud::put(buf, x, area.y, &b.desc, s.desc, right - x);
        }
    }
}
