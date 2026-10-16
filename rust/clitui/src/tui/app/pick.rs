//! The file picker overlay, showing directories and `.urna` files only,
//! themed with the palette. enter walks into a directory or opens the
//! file; esc closes.

use std::path::{Path, PathBuf};

use crossterm::event::{Event, KeyCode};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Clear, Widget};

use crate::tui::pal;
use crate::tui::widgets::Picker;

pub enum Pick {
    Stay,
    Close,
    Open(PathBuf),
}

pub fn new(cwd: &Path) -> anyhow::Result<Picker> {
    let mut p = Picker::new(cwd, |e| {
        e.is_dir || e.path.extension().is_some_and(|x| x == "urna")
    })?;
    land(&mut p);
    Ok(p)
}

/// Routes one input event to the picker.
pub fn handle(fe: &mut Picker, ev: &Event) -> Pick {
    if let Event::Key(k) = ev {
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return Pick::Close,
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Some(f) = fe.selected().filter(|f| f.is_file()) {
                    return Pick::Open(f.path.clone());
                }
            }
            _ => {}
        }
    }
    if fe.handle(ev) {
        land(fe);
    }
    Pick::Stay
}

/// After a directory change, put the cursor on the first `.urna` there,
/// else on the first entry that is not the parent link.
fn land(fe: &mut Picker) {
    let files = fe.files();
    let urna = files.iter().position(|f| f.is_file());
    let first = files.iter().position(|f| f.name != "../");
    if let Some(i) = urna.or(first) {
        fe.set_selected_idx(i);
    }
}

/// The overlay rect for a screen: centered, two thirds of it.
pub fn rect(area: Rect) -> Rect {
    let w = (area.width * 2 / 3).clamp(40.min(area.width), 100);
    let h = (area.height * 2 / 3).clamp(10.min(area.height), 30);
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

pub fn render(buf: &mut Buffer, area: Rect, fe: &Picker) -> Rect {
    // the screen behind the overlay recedes toward the ground.
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            if let Some(c) = buf.cell_mut((x, y)) {
                c.fg = pal::mix(c.fg, pal::BG, 0.65);
            }
        }
    }
    let r = rect(area);
    Clear.render(r, buf);
    buf.set_style(r, Style::new().bg(pal::BG));
    fe.render(buf, r, " open a .urna ");
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_lists_dirs_and_urna_files_only() {
        let d = std::env::temp_dir().join(format!("urna_pick_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("dir")).unwrap();
        std::fs::write(d.join("a.urna"), b"x").unwrap();
        std::fs::write(d.join("b.txt"), b"x").unwrap();
        let fe = new(&d).unwrap();
        let names: Vec<&str> = fe.files().iter().map(|f| f.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("a.urna")));
        assert!(!names.iter().any(|n| n.contains("b.txt")));
        assert_eq!(fe.selected().map(|f| f.name.as_str()), Some("a.urna"));
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        render(&mut buf, area, &fe);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
