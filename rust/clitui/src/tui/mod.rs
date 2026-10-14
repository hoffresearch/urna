//! The terminal ui (feature `tui`, on by default): the interactive
//! installer (`urna setup`) and the corpus explorer (`urna tui`, also what a
//! bare `urna` opens on a terminal). built on ratatui with tachyonfx for
//! every transition; the look is urna.dev's (palette in `pal`, the symbol in
//! `art`), the layout vocabulary is noble's hud (`hud`).
//!
//! nothing here opens a socket: the installer downloads through a `curl`
//! child process, and the explorer only reads local files and spawns the
//! same offline embedder `ask` uses.

pub mod app;
pub mod art;
pub mod fx;
pub mod hud;
pub mod pal;
pub mod rain;
pub mod row;
pub mod setup;
pub mod term;
pub mod toast;
pub mod txt;
