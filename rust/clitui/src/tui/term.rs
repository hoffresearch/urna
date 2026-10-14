//! Terminal lifecycle and the frame clock. Two shapes: the inline viewport
//! the installer draws in (ratatui's inline example: the last frame stays
//! in the scrollback) and the full screen the explorer takes. both restore
//! the terminal on panic before the message prints.

use std::io::{self, Write};
use std::time::{Duration, Instant};

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event};
use crossterm::execute;
use ratatui::{DefaultTerminal, TerminalOptions, Viewport};

/// ~30 fps: smooth for the effects, idle enough for a laptop.
pub const FRAME: Duration = Duration::from_millis(33);

pub fn inline(height: u16) -> io::Result<DefaultTerminal> {
    ratatui::try_init_with_options(TerminalOptions {
        viewport: Viewport::Inline(height),
    })
}

/// Leaves the inline viewport with the cursor on the line below it, so the
/// shell prompt does not land on top of the last frame.
pub fn leave_inline(terminal: &mut DefaultTerminal) -> io::Result<()> {
    let area = terminal.get_frame().area();
    terminal.set_cursor_position((0, area.bottom().saturating_sub(1)))?;
    ratatui::try_restore()?;
    let mut out = io::stdout();
    writeln!(out)?;
    out.flush()
}

pub fn fullscreen() -> io::Result<DefaultTerminal> {
    let terminal = ratatui::try_init()?;
    execute!(io::stdout(), EnableMouseCapture)?;
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = execute!(io::stdout(), DisableMouseCapture);
        prev(info);
    }));
    Ok(terminal)
}

pub fn leave_fullscreen() -> io::Result<()> {
    execute!(io::stdout(), DisableMouseCapture)?;
    ratatui::try_restore()
}

/// Waits up to one frame for input; `None` means "draw the next frame".
pub fn next_event() -> io::Result<Option<Event>> {
    if event::poll(FRAME)? {
        Ok(Some(event::read()?))
    } else {
        Ok(None)
    }
}

/// Wall time between frames, for the effects and the spinners.
pub struct Clock {
    last: Instant,
    pub start: Instant,
}

impl Clock {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last: now,
            start: now,
        }
    }

    /// Time since the previous call.
    pub fn tick(&mut self) -> Duration {
        let now = Instant::now();
        let dt = now - self.last;
        self.last = now;
        dt
    }

    pub fn secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}
