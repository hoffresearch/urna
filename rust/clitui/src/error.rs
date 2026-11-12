//! The cli's error: any error, plus the context each step added on the way
//! up, printed the way a failing `main` has always printed it.
//!
//! `Error` holds the outermost error of a chain; `context` wraps it in a
//! new message whose `source` is the old error, so `{}` prints what went
//! wrong at the top, `{:#}` the whole chain on one line (`a: b: c`), and
//! `{:?}` (what `main` returning `Err` prints) the top line, then
//! `Caused by:` and the chain, numbered when it has more than one cause.
//! `downcast_ref` finds a typed error anywhere in the chain (the explorer
//! reads an `EmbedFailure` that way). `err!` and `bail!` build one from a
//! message.

use std::error::Error as StdError;
use std::fmt::{self, Write as _};

/// Any error with the context added to it.
pub struct Error {
    inner: Box<dyn StdError + Send + Sync + 'static>,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A message with nothing under it.
struct Message(String);

impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for Message {}

/// A context message over the error it explains.
#[derive(Debug)]
struct Wrapped {
    context: String,
    source: Box<dyn StdError + Send + Sync + 'static>,
}

impl fmt::Display for Wrapped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.context)
    }
}

impl StdError for Wrapped {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&*self.source)
    }
}

impl Error {
    pub fn new<E: StdError + Send + Sync + 'static>(error: E) -> Self {
        Self {
            inner: Box::new(error),
        }
    }

    /// An error that is only `message`.
    pub fn msg(message: impl fmt::Display) -> Self {
        Self::new(Message(message.to_string()))
    }

    /// This error under `context`.
    pub fn context(self, context: impl fmt::Display) -> Self {
        Self::new(Wrapped {
            context: context.to_string(),
            source: self.inner,
        })
    }

    /// The error and each of its causes, outermost first.
    pub fn chain(&self) -> impl Iterator<Item = &(dyn StdError + 'static)> {
        let top: &(dyn StdError + 'static) = &*self.inner;
        std::iter::successors(Some(top), |e| e.source())
    }

    /// The first error of type `T` in the chain (the explorer, its only
    /// reader, looks for an `EmbedFailure`).
    #[cfg(feature = "tui")]
    pub fn downcast_ref<T: StdError + 'static>(&self) -> Option<&T> {
        self.chain().find_map(|e| e.downcast_ref::<T>())
    }
}

// no `impl std::error::Error for Error`: it would collide with this blanket
// `From`, which is what lets `?` take any error.
impl<E: StdError + Send + Sync + 'static> From<E> for Error {
    fn from(error: E) -> Self {
        Self::new(error)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)?;
        if f.alternate() {
            for cause in self.chain().skip(1) {
                write!(f, ": {cause}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)?;
        let Some(cause) = self.inner.source() else {
            return Ok(());
        };
        f.write_str("\n\nCaused by:")?;
        let numbered = cause.source().is_some();
        for (n, e) in std::iter::successors(Some(cause), |e| e.source()).enumerate() {
            f.write_char('\n')?;
            let mut out = Indented {
                inner: f,
                number: numbered.then_some(n),
                started: false,
            };
            write!(out, "{e}")?;
        }
        Ok(())
    }
}

/// Writes one cause under `Caused by:`: `    0: ` (or four spaces when
/// there is one cause) before its first line, its next lines aligned.
struct Indented<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
    number: Option<usize>,
    started: bool,
}

impl fmt::Write for Indented<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for (i, line) in s.split('\n').enumerate() {
            if !self.started {
                self.started = true;
                match self.number {
                    Some(n) => write!(self.inner, "{n: >5}: ")?,
                    None => self.inner.write_str("    ")?,
                }
            } else if i > 0 {
                self.inner.write_char('\n')?;
                let pad = if self.number.is_some() {
                    "       "
                } else {
                    "    "
                };
                self.inner.write_str(pad)?;
            }
            self.inner.write_str(line)?;
        }
        Ok(())
    }
}

/// `context` and `with_context` on a `Result` (of any error) or an
/// `Option` (a `None` becomes the context alone).
pub trait Context<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T>;
    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T, E: StdError + Send + Sync + 'static> Context<T> for std::result::Result<T, E> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|e| Error::new(e).context(context))
    }

    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| Error::new(e).context(f()))
    }
}

impl<T> Context<T> for Result<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|e| e.context(context))
    }

    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| e.context(f()))
    }
}

impl<T> Context<T> for Option<T> {
    fn context<C: fmt::Display>(self, context: C) -> Result<T> {
        self.ok_or_else(|| Error::msg(context))
    }

    fn with_context<C: fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| Error::msg(f()))
    }
}

/// An `Error` from a format string (`err!("no {thing}")`) or from one
/// value that displays (`err!(why)`).
macro_rules! err {
    ($msg:literal $(,)?) => {
        $crate::error::Error::msg(format_args!($msg))
    };
    ($msg:expr $(,)?) => {
        $crate::error::Error::msg($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        $crate::error::Error::msg(format_args!($fmt, $($arg)*))
    };
}

/// `return Err(err!(...))`.
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::err!($($arg)*))
    };
}

pub(crate) use {bail, err};

#[cfg(test)]
mod tests {
    use super::*;

    fn io(msg: &str) -> std::io::Error {
        std::io::Error::other(msg.to_string())
    }

    #[test]
    fn prints_like_a_failing_main_always_did() {
        let one: Result<()> = Err(io("no such file")).context("open corpus.urna");
        let one = one.unwrap_err();
        assert_eq!(one.to_string(), "open corpus.urna");
        assert_eq!(format!("{one:#}"), "open corpus.urna: no such file");
        assert_eq!(
            format!("{one:?}"),
            "open corpus.urna\n\nCaused by:\n    no such file"
        );

        let two = one.context("install the payload");
        assert_eq!(
            format!("{two:?}"),
            "install the payload\n\nCaused by:\n    0: open corpus.urna\n    1: no such file"
        );
        let lines = Error::new(io("first\nsecond")).context("top");
        assert_eq!(
            format!("{lines:?}"),
            "top\n\nCaused by:\n    first\n    second"
        );

        let alone = err!("plain {}", 1);
        assert_eq!(format!("{alone:?}"), "plain 1");
        let none: Result<u8> = None.context("nothing there");
        assert_eq!(none.unwrap_err().to_string(), "nothing there");
        fn early(fail: bool) -> Result<()> {
            if fail {
                let why = String::from("a reason");
                bail!(why);
            }
            Ok(())
        }
        assert_eq!(early(true).unwrap_err().to_string(), "a reason");
        assert!(early(false).is_ok());
    }

    #[test]
    #[cfg(feature = "tui")]
    fn finds_a_typed_error_anywhere_in_the_chain() {
        let e = Error::new(io("disk")).context("read").context("load");
        assert_eq!(
            e.downcast_ref::<std::io::Error>().map(|e| e.to_string()),
            Some("disk".into())
        );
        assert!(e.downcast_ref::<std::fmt::Error>().is_none());
    }
}
