//! One frame of the explorer: the chrome, the active tab, the picker
//! overlay, the toasts, then the effects and the color fold.

use std::time::Duration;

use ratatui::DefaultTerminal;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use super::{App, Tab, chrome, corpus, health, hits, home, offer, pick};
use crate::tui::{fx, hud, pal};

impl App {
    pub(super) fn draw(
        &mut self,
        terminal: &mut DefaultTerminal,
        dt: Duration,
    ) -> std::io::Result<()> {
        terminal.draw(|f| {
            let area = f.area();
            self.area = area;
            let buf = f.buffer_mut();
            pal::ground(buf, area);
            if area.width < 50 || area.height < 16 {
                hud::put(
                    buf,
                    area.x + 1,
                    area.y,
                    "urna tui needs at least 50x16",
                    pal::err(),
                    area.width,
                );
                return;
            }
            let right = match (&self.corpus, &self.loading) {
                (_, Some(_)) => format!("{} opening", self.spinner.frame_str()),
                (Some(c), None) => format!(
                    "{} · {}",
                    c.path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    corpus::human(c.size)
                ),
                (None, None) => "no corpus open".into(),
            };
            let (hits, mark) = chrome::header(buf, area, self.tab, &right);
            self.tab_hits = hits;
            let top = chrome::HEADER_ROWS + 1;
            let body = Rect::new(area.x, area.y + top, area.width, area.height - top - 1);
            let footer = Rect::new(area.x, area.bottom() - 1, area.width, 1);
            chrome::footer(
                buf,
                footer,
                self.tab,
                self.corpus.is_some(),
                self.picker.is_some(),
                self.first_run.then_some(self.sample.is_some()),
            );
            let spin = self.spinner.frame_str().to_string();
            match self.tab {
                Tab::Home => {
                    let menu = if self.first_run {
                        home::Menu::First(self.sample.as_deref())
                    } else {
                        home::Menu::Near(&self.files, &self.cwd)
                    };
                    let sym = home::render(buf, body, self.clock.start.elapsed(), &menu);
                    if self.entered {
                        self.fx.add_unique_effect("logo", fx::logo_in(sym));
                        self.fx.add_unique_effect("shimmer", fx::shimmer(sym));
                    }
                }
                Tab::Corpus => match &self.corpus {
                    Some(c) => corpus::render(buf, body, c, self.sec_scroll),
                    None => empty(
                        buf,
                        body,
                        "no corpus open",
                        "o opens the picker; `urna tui file.urna` opens one directly",
                    ),
                },
                Tab::Ask => match &self.corpus {
                    Some(_) => hits::render(buf, body, &mut self.ask, &spin),
                    None => empty(buf, body, "ask needs a corpus", "ctrl+o opens the picker"),
                },
                Tab::Health => health::render(buf, body, &self.health, &spin),
            }
            if self.entered {
                self.entered = false;
                if self.tab != Tab::Home {
                    self.fx.cancel_unique_effect("shimmer");
                }
                self.fx.add_unique_effect("page", fx::page_in(body));
            }
            if self.tab == Tab::Ask
                && let Some(o) = &self.ask.offer
            {
                offer::render(buf, body, o, &spin);
            }
            if let Some(fe) = &self.picker {
                let r = pick::render(buf, area, fe);
                if std::mem::take(&mut self.picker_new) {
                    self.fx.add_unique_effect("picker", fx::open(r));
                }
            }
            for r in self.toasts.render(buf, body) {
                self.fx.add_effect(fx::toast_in(r));
            }
            if !self.booted {
                self.booted = true;
                self.fx.add_effect(fx::boot(area));
                self.fx.add_unique_effect("mark", fx::shimmer(mark));
            }
            self.fx.process_effects(dt, buf, area);
            pal::fit(buf, area, self.depth);
        })?;
        Ok(())
    }
}

fn empty(buf: &mut Buffer, body: Rect, title: &str, hint: &str) {
    let y = body.y + body.height / 3;
    for (i, (s, st)) in [(title, pal::title()), (hint, pal::faint())]
        .into_iter()
        .enumerate()
    {
        let w = (s.chars().count() as u16).min(body.width);
        hud::put(
            buf,
            body.x + (body.width - w) / 2,
            y + i as u16 * 2,
            s,
            st,
            w,
        );
    }
}
