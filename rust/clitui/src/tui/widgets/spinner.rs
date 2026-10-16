//! A braille spinner: ten frames, one every 80 ms of the time the ui
//! feeds it.

use std::time::Duration;

const DOTS: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const STEP: Duration = Duration::from_millis(80);

#[derive(Clone, Debug)]
pub struct Spinner {
    frames: &'static [&'static str],
    at: usize,
    acc: Duration,
}

impl Spinner {
    pub fn dots() -> Self {
        Self {
            frames: &DOTS,
            at: 0,
            acc: Duration::ZERO,
        }
    }

    pub fn tick(&mut self, dt: Duration) {
        self.acc += dt;
        while self.acc >= STEP {
            self.acc -= STEP;
            self.at = (self.at + 1) % self.frames.len();
        }
    }

    pub fn frame_str(&self) -> &'static str {
        self.frames[self.at]
    }
}
