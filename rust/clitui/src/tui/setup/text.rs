//! `urna setup --yes`, and any setup without a terminal on both ends (ci,
//! a pipe, an npm postinstall): the default plan, no questions, one plain
//! line per event, colored only when stdout is a terminal.

use std::io::Write;

use super::job::{self, Ev};
use super::plan::{self, Opts, Task};
use super::scan::Scan;
use crate::cmd::health::Level;
use crate::cmd::tone::{self, Tone};

pub fn run(opts: &Opts) -> i32 {
    let t = Tone::stdout();
    let scan = Scan::probe();
    println!(
        "{} {} {}",
        t.bold("urna setup"),
        scan.version,
        t.fg(&scan.target, tone::FAINT)
    );
    for (label, value, badge) in scan.facts() {
        let color = match badge {
            crate::tui::hud::Badge::Ok => tone::HIGH,
            crate::tui::hud::Badge::Fail => tone::LOW,
            _ => tone::MID,
        };
        println!("  {:<18} {}", label, t.fg(&value, color));
    }
    let items = plan::plan(&scan, opts);
    let tasks: Vec<Task> = items.iter().filter(|i| i.runs()).map(|i| i.task).collect();
    for i in items.iter().filter(|i| !i.runs()) {
        let why = i.blocked.clone().unwrap_or_else(|| i.detail.clone());
        println!(
            "  {} {}: {}",
            t.fg("skip", tone::FAINT),
            i.task.title(),
            why
        );
    }
    let rx = job::spawn(scan, tasks, opts.clone());
    let mut code = plan::blocked_code(&items);
    let mut last_pct = u64::MAX;
    for ev in rx {
        match ev {
            Ev::Start(task) => println!("{} {}", t.fg("==>", tone::TEAL), t.bold(task.title())),
            Ev::Note(_, note) => println!("    {}", t.fg(&note, tone::FAINT)),
            Ev::Bytes(_, n, Some(total)) if total > 0 => {
                let pct = n * 100 / total;
                if pct / 10 != last_pct / 10 || n == total {
                    last_pct = pct;
                    println!(
                        "    {pct:>3}%  {:.1} / {:.1} MB",
                        n as f64 / 1e6,
                        total as f64 / 1e6
                    );
                }
            }
            Ev::Bytes(..) => {}
            Ev::Log(line) => println!("    {}", t.fg(&line, tone::FAINT)),
            Ev::Health(rows) => {
                for r in rows {
                    let tag = match r.level {
                        Level::Ok => t.fg("ok      ", tone::HIGH),
                        Level::Warn => t.fg("warn    ", tone::MID),
                        Level::Fail(c) => t.fg(&format!("fail({c})"), tone::LOW),
                    };
                    println!("    {tag} {}: {}", r.what, r.detail);
                }
            }
            Ev::Done(task, Ok(msg)) => {
                println!("  {} {}: {msg}", t.fg("ok", tone::HIGH), task.title())
            }
            Ev::Done(task, Err((c, msg))) => {
                println!(
                    "  {} {}: {msg}",
                    t.fg(&format!("fail({c})"), tone::LOW),
                    task.title()
                );
                if code == 0 {
                    code = c;
                }
            }
            Ev::End => {}
        }
        let _ = std::io::stdout().flush();
    }
    if code == 0 {
        println!("{} {}", t.bold("urna setup:"), t.fg("ready", tone::HIGH));
        println!("  try: urna tui   ·   urna ask corpus.urna \"...\"   ·   https://urna.dev");
        println!("  to build a corpus: urna build --spec corpus.toml, from a checkout of the repo");
    } else {
        println!("{} {}", t.bold("urna setup:"), t.fg("failed", tone::LOW));
    }
    code
}
