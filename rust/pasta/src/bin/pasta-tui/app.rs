//! The TUI's state: what was typed, the list under it, the selected card
//! and its image. Typing refreshes the suggestions; Enter searches.

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use pasta_core::{Filter, Pasta};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The list is the type-ahead for what is in the input.
    Suggest,
    /// The list is the result of the last Enter.
    Results,
    /// Nothing typed: the most played cards.
    Browse,
}

pub struct Row {
    pub ordinal: usize,
    /// The score of a search hit; `None` for suggestions and browsing.
    pub score: Option<f32>,
}

pub struct Shown {
    pub ordinal: usize,
    pub image: Option<StatefulProtocol>,
    /// the decoded pixels, for the quadrant renderer.
    pub rgba: Option<image::RgbaImage>,
    /// the quadrant cells last drawn, and the rect they were drawn for.
    pub blocks: Option<(ratatui::layout::Rect, Vec<crate::blocks::Cell>)>,
    pub decode_ms: f64,
    pub error: Option<String>,
}

pub struct App<'a> {
    pub pasta: &'a Pasta,
    pub picker: Picker,
    pub input: String,
    pub mode: Mode,
    pub rows: Vec<Row>,
    pub sel: usize,
    pub shown: Option<Shown>,
    pub last_ms: f64,
    pub quit: bool,
}

impl<'a> App<'a> {
    pub fn new(pasta: &'a Pasta, picker: Picker) -> Self {
        let mut app = Self {
            pasta,
            picker,
            input: String::new(),
            mode: Mode::Browse,
            rows: Vec::new(),
            sel: 0,
            shown: None,
            last_ms: 0.0,
            quit: false,
        };
        app.refresh();
        app
    }

    /// The text before `|` and the filters after it.
    fn split_input(&self) -> (String, Filter) {
        let (q, spec) = self.input.split_once('|').unwrap_or((&self.input, ""));
        (q.trim().to_string(), Filter::parse(spec))
    }

    /// Recompute the list from the input: suggestions while typing a name,
    /// popular cards under the filters when there is no text.
    fn refresh(&mut self) {
        let t = Instant::now();
        let (q, filter) = self.split_input();
        if q.is_empty() {
            self.mode = Mode::Browse;
            let hits = self.pasta.search("", &filter, 60).unwrap_or_default();
            self.rows = hits
                .into_iter()
                .map(|h| Row {
                    ordinal: h.ordinal,
                    score: None,
                })
                .collect();
        } else {
            self.mode = Mode::Suggest;
            self.rows = self
                .pasta
                .complete(&q, 30)
                .into_iter()
                .filter(|s| filter.keeps(&self.pasta.cards()[s.ordinal]))
                .map(|s| Row {
                    ordinal: s.ordinal,
                    score: None,
                })
                .collect();
        }
        self.last_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.sel = 0;
    }

    fn search(&mut self) {
        let t = Instant::now();
        let (q, filter) = self.split_input();
        if let Ok(hits) = self.pasta.search(&q, &filter, 60) {
            self.rows = hits
                .into_iter()
                .map(|h| Row {
                    ordinal: h.ordinal,
                    score: Some(h.score),
                })
                .collect();
        }
        self.mode = if q.is_empty() {
            Mode::Browse
        } else {
            Mode::Results
        };
        self.last_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.sel = 0;
    }

    pub fn selected(&self) -> Option<usize> {
        self.rows.get(self.sel).map(|r| r.ordinal)
    }

    /// Decode the selected card's image when the selection moved.
    pub fn sync_image(&mut self) {
        let Some(ordinal) = self.selected() else {
            self.shown = None;
            return;
        };
        if self.shown.as_ref().is_some_and(|s| s.ordinal == ordinal) {
            return;
        }
        let t = Instant::now();
        let (image, rgba, error) = match self.pasta.decode_image(ordinal) {
            Ok(img) => (
                Some(self.picker.new_resize_protocol(img.clone())),
                Some(img.to_rgba8()),
                None,
            ),
            Err(e) => (None, None, Some(e.to_string())),
        };
        let decode_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.shown = Some(Shown {
            ordinal,
            image,
            rgba,
            blocks: None,
            decode_ms,
            error,
        });
    }

    pub fn key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Char('c') if ctrl => self.quit = true,
            KeyCode::Esc if self.input.is_empty() => self.quit = true,
            KeyCode::Esc => {
                self.input.clear();
                self.refresh();
            }
            KeyCode::Char('u') if ctrl => {
                self.input.clear();
                self.refresh();
            }
            KeyCode::Char(c) => {
                self.input.push(c);
                self.refresh();
            }
            KeyCode::Backspace => {
                self.input.pop();
                self.refresh();
            }
            KeyCode::Enter => self.search(),
            KeyCode::Tab => {
                if let Some(o) = self.selected() {
                    let filters = self
                        .input
                        .split_once('|')
                        .map(|(_, f)| format!(" |{f}"))
                        .unwrap_or_default();
                    self.input = format!("{}{}", self.pasta.cards()[o].name, filters);
                    self.search();
                }
            }
            KeyCode::Down => self.sel = (self.sel + 1).min(self.rows.len().saturating_sub(1)),
            KeyCode::Up => self.sel = self.sel.saturating_sub(1),
            KeyCode::PageDown => self.sel = (self.sel + 10).min(self.rows.len().saturating_sub(1)),
            KeyCode::PageUp => self.sel = self.sel.saturating_sub(10),
            _ => {}
        }
    }
}
