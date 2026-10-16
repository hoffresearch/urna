//! The file picker: a directory listing the caller filters (directories
//! and `.urna` files), a cursor, and the keys that walk it. Enter or the
//! right arrow walk into a directory, the left arrow goes up, ctrl+h shows
//! hidden entries. The caller decides what enter does on a file.

use std::path::{Path, PathBuf};

use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::widgets::{Block, BorderType, Borders, Widget};

use crate::tui::{hud, pal};

#[derive(Clone, Debug)]
pub struct Entry {
    /// the name as shown: directories end in `/`, the parent is `../`.
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
}

impl Entry {
    pub fn is_file(&self) -> bool {
        !self.is_dir
    }
}

type Keep = Box<dyn Fn(&Entry) -> bool + Send>;

pub struct Picker {
    cwd: PathBuf,
    entries: Vec<Entry>,
    selected: usize,
    hidden: bool,
    keep: Keep,
}

impl std::fmt::Debug for Picker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Picker")
            .field("cwd", &self.cwd)
            .field("entries", &self.entries.len())
            .field("selected", &self.selected)
            .finish_non_exhaustive()
    }
}

impl Picker {
    /// List `cwd` with `keep` deciding which entries show (the parent link
    /// always does).
    pub fn new(
        cwd: &Path,
        keep: impl Fn(&Entry) -> bool + Send + 'static,
    ) -> std::io::Result<Self> {
        let mut p = Self {
            cwd: cwd.to_path_buf(),
            entries: Vec::new(),
            selected: 0,
            hidden: false,
            keep: Box::new(keep),
        };
        p.reload()?;
        Ok(p)
    }

    fn reload(&mut self) -> std::io::Result<()> {
        let mut dirs = Vec::new();
        let mut files = Vec::new();
        for e in std::fs::read_dir(&self.cwd)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if !self.hidden && name.starts_with('.') {
                continue;
            }
            let is_dir = e.file_type().is_ok_and(|t| t.is_dir());
            let entry = Entry {
                name: if is_dir { format!("{name}/") } else { name },
                path: e.path(),
                is_dir,
            };
            if (self.keep)(&entry) {
                if is_dir { &mut dirs } else { &mut files }.push(entry);
            }
        }
        dirs.sort_by_key(|e| e.name.to_lowercase());
        files.sort_by_key(|e| e.name.to_lowercase());
        let mut entries = Vec::with_capacity(dirs.len() + files.len() + 1);
        if let Some(parent) = self.cwd.parent() {
            entries.push(Entry {
                name: "../".into(),
                path: parent.to_path_buf(),
                is_dir: true,
            });
        }
        entries.extend(dirs);
        entries.extend(files);
        self.entries = entries;
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        Ok(())
    }

    pub fn files(&self) -> &[Entry] {
        &self.entries
    }

    pub fn set_selected_idx(&mut self, i: usize) {
        self.selected = i.min(self.entries.len().saturating_sub(1));
    }

    pub fn selected(&self) -> Option<&Entry> {
        self.entries.get(self.selected)
    }

    /// Walk into `dir`; the listing is kept when it cannot be read.
    pub fn enter(&mut self, dir: &Path) {
        let before = std::mem::replace(&mut self.cwd, dir.to_path_buf());
        self.selected = 0;
        if self.reload().is_err() {
            self.cwd = before;
            let _ = self.reload();
        }
    }

    /// The keys the picker handles itself; `true` when the directory
    /// changed.
    pub fn handle(&mut self, ev: &Event) -> bool {
        let Event::Key(k) = ev else {
            return false;
        };
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.set_selected_idx(self.selected + 1),
            KeyCode::PageUp => self.selected = self.selected.saturating_sub(10),
            KeyCode::PageDown => self.set_selected_idx(self.selected + 10),
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.set_selected_idx(usize::MAX),
            KeyCode::Char('h') if ctrl => {
                self.hidden = !self.hidden;
                let _ = self.reload();
            }
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Backspace => {
                if let Some(parent) = self.cwd.parent().map(Path::to_path_buf) {
                    self.enter(&parent);
                    return true;
                }
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Some(e) = self.selected().filter(|e| e.is_dir).cloned() {
                    self.enter(&e.path);
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    /// Draw the listing in a rounded panel titled `title`, the directory
    /// along the bottom border, the cursor row highlighted.
    pub fn render(&self, buf: &mut Buffer, area: Rect, title: &str) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(pal::GLOW))
            .title(title)
            .title_style(pal::title())
            .title_bottom(format!(" {} ", self.cwd.display()))
            .style(Style::new().fg(pal::INK).bg(pal::BG));
        let inner = block.inner(area);
        block.render(area, buf);
        let rows = usize::from(inner.height);
        if rows == 0 || inner.width == 0 {
            return;
        }
        let first = self.selected.saturating_sub(rows - 1);
        for (i, e) in self.entries.iter().enumerate().skip(first).take(rows) {
            let y = inner.y + (i - first) as u16;
            let sel = i == self.selected;
            let style = match (sel, e.is_dir) {
                (true, true) => Style::new()
                    .fg(pal::TITLE_FROM)
                    .bg(pal::SURFACE)
                    .add_modifier(Modifier::BOLD),
                (true, false) => Style::new()
                    .fg(pal::INK_HI)
                    .bg(pal::SURFACE)
                    .add_modifier(Modifier::BOLD),
                (false, true) => pal::accent(),
                (false, false) => Style::new().fg(pal::INK),
            };
            if sel {
                buf.set_style(
                    Rect::new(inner.x, y, inner.width, 1),
                    Style::new().bg(pal::SURFACE),
                );
            }
            let x = hud::put(
                buf,
                inner.x,
                y,
                if sel { "› " } else { "  " },
                pal::key(),
                inner.width,
            );
            hud::put(buf, x, y, &e.name, style, inner.right().saturating_sub(x));
        }
    }
}
