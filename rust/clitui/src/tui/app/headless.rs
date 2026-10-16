//! The ask tab without a terminal, for checking the screen and the
//! type-ahead from a script: `--snapshot QUERY` asks and prints one frame
//! as text; `--eval N` types N ranked chunks' labels key by key and
//! searches each, and reports where the card lands and how long each
//! keystroke took. Both need chunk_scalars with label and rank columns.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui_image::picker::{Picker, ProtocolType};

use super::ask::{self, Ask};
use super::session::Session;
use super::{corpus, hits};

const COLS: u16 = 120;
const ROWS: u16 = 40;

/// Opens `file` for the headless modes; without labels in the file it says so
/// and the caller exits 2.
fn open(file: &Path) -> Result<Result<Arc<Session>, i32>> {
    let c = corpus::load(file)?;
    if c.session.label(0).is_none() && c.session.scalars().is_none() {
        eprintln!(
            "{}: the corpus carries no chunk_scalars with a label column (see `urna scalars`)",
            file.display()
        );
        return Ok(Err(2));
    }
    Ok(Ok(c.session))
}

/// Asks `query` and waits for the answer, up to a minute.
fn ask_and_wait(ask: &mut Ask, session: Arc<Session>, query: &str) -> Result<(), String> {
    ask.input.set_value(query.to_string());
    ask.input.end();
    ask.submit(session);
    let t0 = Instant::now();
    while ask.busy() {
        if let Some(reply) = ask.poll() {
            return reply.map(|_| ()).map_err(|f| f.text);
        }
        if t0.elapsed() > Duration::from_secs(60) {
            return Err("the query did not answer in a minute".into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

pub fn snapshot(file: &Path, query: &str) -> Result<i32> {
    let session = match open(file)? {
        Ok(s) => s,
        Err(code) => return Ok(code),
    };
    let mut ask = Ask::default();
    let mut picker = Picker::halfblocks();
    picker.set_protocol_type(ProtocolType::Halfblocks);
    ask.media.set_picker(picker);
    if let Err(e) = ask_and_wait(&mut ask, session, query) {
        eprintln!("ask failed: {e}");
        return Ok(1);
    }
    let area = Rect::new(0, 0, COLS, ROWS);
    let mut buf = Buffer::empty(area);
    hits::render(&mut buf, area, &mut ask, "·");
    for y in 0..ROWS {
        let line: String = (0..COLS)
            .map(|x| buf[(x, y)].symbol().to_string())
            .collect();
        println!("{}", line.trim_end());
    }
    Ok(0)
}

/// The label of `name` in the suggestions: the name itself, or the name
/// with its Portuguese alias.
fn is_label(label: &str, name: &str) -> bool {
    label == name
        || label
            .strip_prefix(name)
            .is_some_and(|r| r.starts_with(" ("))
}

pub fn eval(file: &Path, want: usize) -> Result<i32> {
    let session = match open(file)? {
        Ok(s) => s,
        Err(code) => return Ok(code),
    };
    let n_chunks = session.rt.chunk_ids().len();
    let all: Vec<(usize, String)> = (0..n_chunks)
        .filter(|&i| session.rank(i).is_some())
        .filter_map(|i| session.label(i).map(|l| (i, l.to_string())))
        .collect();
    let want = want.max(1);
    let stride = (all.len() / want).max(1);
    let names: Vec<(usize, String)> = all.into_iter().step_by(stride).collect();
    let keep = |_: usize| true;
    let (mut full_top1, mut search_top1, mut keys_to_top5) = (0usize, 0usize, Vec::new());
    let mut lat = Vec::new();
    for (ord, name) in &names {
        let chars: Vec<char> = name.chars().collect();
        let mut found = None;
        for n in 1..=chars.len() {
            let prefix: String = chars[..n].iter().collect();
            let t = Instant::now();
            let s = session.labels.suggest_where(&prefix, 5, &keep);
            lat.push(t.elapsed().as_secs_f64() * 1000.0);
            if s.iter().any(|x| is_label(x, name)) {
                found = Some(n);
                break;
            }
        }
        keys_to_top5.push(found.unwrap_or(usize::MAX));
        if session
            .labels
            .suggest_where(name, 1, &keep)
            .first()
            .is_some_and(|x| is_label(x, name))
        {
            full_top1 += 1;
        }
        let hits = match ask::run_query(&session, name, None) {
            Ok(h) => h,
            Err(f) => {
                eprintln!("search failed: {}", f.text);
                return Ok(1);
            }
        };
        if hits
            .first()
            .is_some_and(|h| h.chunk == *ord || session.label(h.chunk) == Some(name.as_str()))
        {
            search_top1 += 1;
        }
    }
    lat.sort_by(f64::total_cmp);
    let n = names.len().max(1);
    let mut reached: Vec<usize> = keys_to_top5
        .iter()
        .copied()
        .filter(|k| *k != usize::MAX)
        .collect();
    reached.sort_unstable();
    let at = |v: &[usize], i: usize| v.get(i).copied().unwrap_or(0);
    let lat_at = |i: usize| lat.get(i).copied().unwrap_or(0.0);
    println!("ranked chunks evaluated: {n} (stride {stride})");
    println!(
        "full name -> 1st suggestion: {full_top1} ({:.4})",
        full_top1 as f64 / n as f64
    );
    println!(
        "name search -> 1st hit: {search_top1} ({:.4})",
        search_top1 as f64 / n as f64
    );
    println!(
        "keystrokes until the card is in the top 5: median {} p90 {} never {}",
        at(&reached, reached.len() / 2),
        at(&reached, reached.len() * 9 / 10),
        n - reached.len()
    );
    println!(
        "keystroke latency over {} calls: p50 {:.2} ms p99 {:.2} ms max {:.2} ms",
        lat.len(),
        lat_at(lat.len() / 2),
        lat_at(lat.len() * 99 / 100),
        lat_at(lat.len().saturating_sub(1))
    );
    Ok(0)
}
