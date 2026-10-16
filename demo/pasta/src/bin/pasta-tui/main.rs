//! `pasta-tui FILE.urna FIELDS POTION_DIR [--graphics auto|halfblocks|sixel|kitty|iterm]`
//!
//! Type a card name and the suggestions follow each key; Enter searches
//! names, rules text and filters; the selected card shows its image from
//! the `.urna`, decoded in process. `--snapshot QUERY` draws one frame of
//! QUERY into text and exits, for checking the screen without a terminal.

mod app;
mod blocks;
mod view;

use std::path::PathBuf;
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use pasta_core::Pasta;
use ratatui_image::picker::{Picker, ProtocolType};

fn forced(choice: &str) -> Option<ProtocolType> {
    match choice {
        "halfblocks" => Some(ProtocolType::Halfblocks),
        "sixel" => Some(ProtocolType::Sixel),
        "kitty" => Some(ProtocolType::Kitty),
        "iterm" => Some(ProtocolType::Iterm2),
        _ => None,
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let paths: Vec<PathBuf> = args.iter().skip(1).filter(|a| !a.starts_with("--")).take(3).map(PathBuf::from).collect();
    if paths.len() != 3 {
        eprintln!("usage: pasta-tui FILE.urna FIELDS POTION_DIR [--graphics auto|halfblocks|sixel|kitty|iterm] [--snapshot QUERY]");
        std::process::exit(2);
    }
    eprintln!("abrindo {} …", paths[0].display());
    let pasta = match Pasta::open(&paths[0], &paths[1], &paths[2]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    if let Some(query) = flag(&args, "--snapshot") {
        snapshot(&pasta, &query);
        return;
    }

    let mut terminal = ratatui::init();
    // the picker asks the terminal what it can draw; it must run in raw mode
    // and before anything else reads stdin.
    let mut picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    if let Some(p) = flag(&args, "--graphics").as_deref().and_then(forced) {
        picker.set_protocol_type(p);
    }
    let mut app = app::App::new(&pasta, picker);
    let result = run(&mut terminal, &mut app);
    ratatui::restore();
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut app::App) -> std::io::Result<()> {
    while !app.quit {
        app.sync_image();
        terminal.draw(|f| view::draw(f, app))?;
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        // drain what is queued, so fast typing redraws once.
        loop {
            if let Event::Key(k) = event::read()?
                && k.kind == KeyEventKind::Press
            {
                app.key(k);
            }
            if app.quit || !event::poll(Duration::ZERO)? {
                break;
            }
        }
    }
    Ok(())
}

/// Type QUERY, press Enter, draw one 120x40 frame with halfblocks into a
/// buffer and print it as text.
fn snapshot(pasta: &Pasta, query: &str) {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let mut picker = Picker::halfblocks();
    picker.set_protocol_type(ProtocolType::Halfblocks);
    let mut app = app::App::new(pasta, picker);
    for c in query.chars() {
        app.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
    }
    app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    app.sync_image();
    let backend = ratatui::backend::TestBackend::new(120, 40);
    let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
    terminal.draw(|f| view::draw(f, &mut app)).expect("draw");
    let buf = terminal.backend().buffer();
    for y in 0..buf.area.height {
        let line: String = (0..buf.area.width).map(|x| buf[(x, y)].symbol().to_string()).collect();
        println!("{line}");
    }
}
