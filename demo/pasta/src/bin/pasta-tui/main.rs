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

/// The cell's pixel size from the window size the kernel reports.
fn cell_size() -> ratatui_image::FontSize {
    crossterm::terminal::window_size()
        .ok()
        .filter(|w| w.width > 0 && w.height > 0 && w.columns > 0 && w.rows > 0)
        .map(|w| ratatui_image::FontSize { width: w.width / w.columns, height: w.height / w.rows })
        .unwrap_or(ratatui_image::FontSize { width: 10, height: 20 })
}

/// Kitty's protocol in kitty and ghostty, iTerm2's in iTerm2 and WezTerm,
/// characters everywhere else (and inside tmux).
fn from_env() -> ProtocolType {
    let var = |k: &str| std::env::var(k).unwrap_or_default();
    let (term, program) = (var("TERM"), var("TERM_PROGRAM"));
    if term.starts_with("tmux") || term.starts_with("screen") {
        ProtocolType::Halfblocks
    } else if !var("KITTY_WINDOW_ID").is_empty() || term.contains("kitty") || program == "ghostty" {
        ProtocolType::Kitty
    } else if program == "iTerm.app" || program == "WezTerm" || var("LC_TERMINAL").contains("iTerm") {
        ProtocolType::Iterm2
    } else {
        ProtocolType::Halfblocks
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
    // decided without querying the terminal: a query's reply is read from
    // stdin, and a reply that never comes takes the first keys typed.
    // deprecated upstream in favor of the stdin query, the one thing this
    // avoids; it is the constructor that takes a known cell size.
    #[allow(deprecated)]
    let mut picker = Picker::from_fontsize(cell_size());
    picker.set_protocol_type(flag(&args, "--graphics").as_deref().and_then(forced).unwrap_or_else(from_env));
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
