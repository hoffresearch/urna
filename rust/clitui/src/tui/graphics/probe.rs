//! The terminal asked, once, what it draws. Four queries go out in one
//! write and the replies are read off stdin with `poll` and a deadline,
//! on the calling thread, before the first key is read:
//!
//! - kitty's graphics query (`a=q`): `OK` when the terminal takes kitty
//!   images;
//! - XTVERSION (`CSI > q`): the terminal's name, which tells iTerm2 and
//!   WezTerm (inline images) and tmux apart;
//! - `CSI 16 t`: the cell size in pixels;
//! - DA1 (`CSI c`), last: every terminal answers it, and in order, so its
//!   reply closes the read; attribute 4 is sixel.
//!
//! A terminal that answers nothing costs the deadline and keeps nothing
//! reading stdin after it. The read takes one byte at a time, so a key
//! typed after DA1 stays for the explorer; replies that come after the
//! deadline (a slow link) are taken out of the key stream by `Late`.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crossterm::event::{Event, KeyCode, KeyModifiers};

/// The longest the replies are waited for; DA1 ends the wait sooner on
/// every terminal that answers.
pub const DEADLINE: Duration = Duration::from_millis(1000);

/// When `ask` stopped before DA1: the replies may still be on their way.
static GAVE_UP: OnceLock<Instant> = OnceLock::new();

/// How long after the deadline a late reply is still looked for.
const LATE: Duration = Duration::from_secs(10);

/// A reply arrives in one burst: one still open after this was a key.
const BURST: Duration = Duration::from_millis(100);

/// What the terminal said; `None` where it said nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Answer {
    pub kitty: bool,
    pub sixel: bool,
    /// (width, height) of a cell in pixels.
    pub cell: Option<(u16, u16)>,
    /// XTVERSION's text, such as `kitty(0.39.1)` or `tmux 3.5a`.
    pub name: Option<String>,
    /// DA1 came back: the terminal answered, and in full.
    pub done: bool,
}

/// The four queries, in the order the replies close the read. The query,
/// and the parser of its replies, exist where a terminal is asked (unix).
#[cfg(any(unix, test))]
pub fn query() -> Vec<u8> {
    const KITTY: &[u8] = b"\x1b_Gi=31,s=1,v=1,a=q,t=d,f=24;AAAA\x1b\\";
    const XTVERSION: &[u8] = b"\x1b[>0q";
    const CELL: &[u8] = b"\x1b[16t";
    const DA1: &[u8] = b"\x1b[c";
    [KITTY, XTVERSION, CELL, DA1].concat()
}

/// The replies in `bytes`; what is not a reply (a key, a mouse report) is
/// skipped.
#[cfg(any(unix, test))]
pub fn parse(bytes: &[u8]) -> Answer {
    let mut a = Answer::default();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != 0x1b {
            i += 1;
            continue;
        }
        let rest = &bytes[i..];
        if let Some(body) = rest.strip_prefix(b"\x1b_G") {
            // APC: `i=31;OK` or `i=31;ENOTSUPPORTED:...`, up to ST.
            let Some(end) = find(body, b"\x1b\\") else {
                break;
            };
            let reply = &body[..end];
            if reply.starts_with(b"i=31;") && reply.ends_with(b";OK") {
                a.kitty = true;
            }
            i += 3 + end + 2;
        } else if let Some(body) = rest.strip_prefix(b"\x1bP>|") {
            // DCS: XTVERSION's text, up to ST.
            let Some(end) = find(body, b"\x1b\\") else {
                break;
            };
            a.name = Some(String::from_utf8_lossy(&body[..end]).trim().to_string());
            i += 4 + end + 2;
        } else if let Some(body) = rest.strip_prefix(b"\x1b[") {
            // CSI: parameters up to the final byte.
            let Some(end) = body.iter().position(|b| (0x40..=0x7e).contains(b)) else {
                break;
            };
            csi(&body[..end], body[end], &mut a);
            i += 2 + end + 1;
        } else {
            i += 1;
        }
    }
    a
}

#[cfg(any(unix, test))]
fn csi(params: &[u8], fin: u8, a: &mut Answer) {
    let text = String::from_utf8_lossy(params);
    match fin {
        // DA1: `?62;4;22`; the first number is the class, the rest are
        // attributes, 4 among them where sixel is drawn.
        b'c' if text.starts_with('?') => {
            let attrs: Vec<&str> = text[1..].split(';').collect();
            a.sixel = attrs.iter().skip(1).any(|p| *p == "4");
            a.done = true;
        }
        // `CSI 6 ; height ; width t`: the cell in pixels.
        b't' => {
            let n: Vec<u16> = text.split(';').filter_map(|p| p.parse().ok()).collect();
            if let [6, h, w] = n[..]
                && w > 0
                && h > 0
            {
                a.cell = Some((w, h));
            }
        }
        _ => {}
    }
}

#[cfg(any(unix, test))]
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Ask the terminal on stdin/stdout, which must be the raw-mode tty the
/// explorer draws on; `None` when stdin is not a terminal. The read stops
/// at DA1's reply or at `deadline`, whichever comes first.
#[cfg(unix)]
pub fn ask(deadline: Duration) -> Option<Answer> {
    use std::io::{IsTerminal, Write};
    use std::os::fd::AsRawFd;

    let stdin = std::io::stdin();
    if !stdin.is_terminal() {
        return None;
    }
    let mut out = std::io::stdout();
    out.write_all(&query()).ok()?;
    out.flush().ok()?;
    let fd = stdin.as_raw_fd();
    let until = Instant::now() + deadline;
    let mut got = Vec::new();
    // one byte per read: nothing past DA1 leaves the kernel's queue.
    let mut byte = 0u8;
    loop {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        let mut p = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ms = i32::try_from(left.as_millis().max(1)).unwrap_or(i32::MAX);
        // SAFETY: one valid pollfd for the length given.
        let ready = unsafe { libc::poll(&mut p, 1, ms) };
        if ready < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        if ready == 0 {
            break;
        }
        // SAFETY: `byte` is valid for one byte; poll said fd reads.
        let n = unsafe { libc::read(fd, (&raw mut byte).cast(), 1) };
        if n <= 0 {
            break;
        }
        got.push(byte);
        if byte == b'c' && parse(&got).done {
            break;
        }
    }
    let answer = parse(&got);
    if !answer.done {
        let _ = GAVE_UP.set(Instant::now());
    }
    Some(answer)
}

/// Windows reads the console as input records, not bytes: no query there.
#[cfg(not(unix))]
pub fn ask(_deadline: Duration) -> Option<Answer> {
    None
}

/// The kitty and XTVERSION replies that arrive after `ask` gave up, as
/// crossterm reads them: `Alt+_` (APC) or `Alt+P` (DCS), the reply's
/// characters as keys, `Alt+\` (ST). DA1 and the cell size are CSI, which
/// crossterm drops itself.
#[derive(Default)]
pub struct Late {
    open: Option<Instant>,
}

impl Late {
    /// `ev` is part of a late reply, not a key.
    pub fn swallows(&mut self, ev: &Event, now: Instant) -> bool {
        let armed = GAVE_UP
            .get()
            .is_some_and(|t| now.saturating_duration_since(*t) < LATE);
        self.swallows_when(armed, ev, now)
    }

    fn swallows_when(&mut self, armed: bool, ev: &Event, now: Instant) -> bool {
        let Event::Key(k) = ev else { return false };
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        if let Some(t) = self.open {
            if now.saturating_duration_since(t) <= BURST {
                if alt && k.code == KeyCode::Char('\\') {
                    self.open = None;
                }
                return true;
            }
            self.open = None;
        }
        if armed && alt && matches!(k.code, KeyCode::Char('_' | 'P')) {
            self.open = Some(now);
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitty_answers_ok_its_name_its_cell_and_da1() {
        let a = parse(b"\x1b_Gi=31;OK\x1b\\\x1bP>|kitty(0.39.1)\x1b\\\x1b[6;20;10t\x1b[?62;c");
        assert!(a.kitty && a.done && !a.sixel);
        assert_eq!(a.cell, Some((10, 20)));
        assert_eq!(a.name.as_deref(), Some("kitty(0.39.1)"));
    }

    #[test]
    fn a_sixel_terminal_says_4_after_the_class() {
        let a = parse(b"\x1bP>|foot(1.18.1)\x1b\\\x1b[6;17;8t\x1b[?62;4;22c");
        assert!(a.sixel && !a.kitty && a.done);
        assert_eq!(a.cell, Some((8, 17)));
        // `?1;2c` (a VT100) has no attribute 4; the class 4 is not one.
        assert!(!parse(b"\x1b[?1;2c").sixel);
        assert!(!parse(b"\x1b[?4;6c").sixel);
    }

    #[test]
    fn a_refused_kitty_query_a_key_and_a_mouse_report_are_not_replies() {
        let a = parse(b"x\x1b_Gi=31;ENOTSUPPORTED:no\x1b\\\x1b[<0;3;4M\x1b[?65;1;9c");
        assert!(!a.kitty && !a.sixel && a.done);
        assert_eq!(a.cell, None);
    }

    #[test]
    fn a_reply_cut_short_waits_for_the_rest() {
        let whole = b"\x1bP>|WezTerm 20240203\x1b\\\x1b[?65;4c";
        assert!(!parse(&whole[..9]).done);
        let a = parse(whole);
        assert_eq!(a.name.as_deref(), Some("WezTerm 20240203"));
        assert!(a.done && a.sixel);
    }

    #[test]
    fn a_late_reply_is_not_typed_and_a_key_after_it_is() {
        use crossterm::event::KeyEvent;
        let key = |c: char, m: KeyModifiers| Event::Key(KeyEvent::new(KeyCode::Char(c), m));
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut late = Late::default();
        // `\x1b_Gi=31;OK\x1b\\` as crossterm reads it.
        assert!(late.swallows_when(true, &key('_', KeyModifiers::ALT), at(0)));
        for c in "Gi=31;OK".chars() {
            assert!(late.swallows_when(true, &key(c, KeyModifiers::NONE), at(1)));
        }
        assert!(late.swallows_when(true, &key('\\', KeyModifiers::ALT), at(1)));
        assert!(!late.swallows_when(true, &key('a', KeyModifiers::NONE), at(2)));
        // an Alt+P that no reply follows lets the next key through.
        assert!(late.swallows_when(
            true,
            &key('P', KeyModifiers::ALT | KeyModifiers::SHIFT),
            at(10)
        ));
        assert!(!late.swallows_when(true, &key('b', KeyModifiers::NONE), at(500)));
        // disarmed (DA1 came back, or long after): Alt+_ is a key.
        assert!(!late.swallows_when(false, &key('_', KeyModifiers::ALT), at(600)));
    }

    #[test]
    fn the_query_ends_with_da1() {
        assert!(query().ends_with(b"\x1b[c"));
        assert!(query().starts_with(b"\x1b_G"));
    }
}
