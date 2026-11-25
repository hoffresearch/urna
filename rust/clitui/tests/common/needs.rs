//! The needs of a test: a named external resource it can lack (a python
//! with the forge's packages, the git-lfs files, a corpus). The contract is
//! tool/tests/testneeds.py's, over the same table (testneeds.tsv):
//!
//! - the run requires the need (`URNA_REQUIRE_<NEED>=1` or
//!   `URNA_REQUIRE_ALL=1`): the test fails, naming the need and how to
//!   provide it;
//! - otherwise it prints `skip: <need>: <test>: <reason>`, appends
//!   `<need>\t<test>\t<reason>` to `URNA_SKIP_LOG` when that names a file,
//!   and the test returns.
//!
//! Not a crate: every test that can lack a need includes this file with
//! `#[path = ...] mod needs;`, the parity workspaces too, so nothing here is
//! public API and nothing depends on more than std.

use std::ffi::OsString;
use std::io::Write;

const TABLE: &str = include_str!("../../../../tool/tests/testneeds.tsv");

/// How to provide `need`, or `None` when the table does not list it.
fn provide(need: &str) -> Option<&'static str> {
    TABLE
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('\t'))
        .find(|(n, _)| *n == need)
        .map(|(_, how)| how)
}

/// The name of the running test: libtest names each test's thread after it.
fn test_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("an unnamed test")
        .to_owned()
}

/// The test lacks `need`, for `reason`: a panic when the run requires it,
/// else the skip line and the log line. The caller returns after it.
pub fn lack(need: &str, reason: &str) {
    decide(need, &test_name(), reason, &|k| std::env::var_os(k));
}

/// `lack` with the test's name and the environment passed in, so its own
/// test needs no global state. Panics on a need the table does not list.
pub fn decide(need: &str, test: &str, reason: &str, env: &dyn Fn(&str) -> Option<OsString>) {
    let Some(how) = provide(need) else {
        panic!("{need}: not a need of tool/tests/testneeds.tsv");
    };
    let var = format!("URNA_REQUIRE_{}", need.to_uppercase());
    let on = |k: &str| env(k).is_some_and(|v| v == "1");
    if on(&var) || on("URNA_REQUIRE_ALL") {
        panic!("{need} is required ({var} or URNA_REQUIRE_ALL): {test}: {reason}; provide {how}");
    }
    eprintln!("skip: {need}: {test}: {reason}");
    if let Some(log) = env("URNA_SKIP_LOG").filter(|p| !p.is_empty()) {
        let line = format!("{need}\t{test}\t{reason}\n");
        if let Err(e) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .and_then(|mut f| f.write_all(line.as_bytes()))
        {
            panic!("URNA_SKIP_LOG {}: {e}", log.to_string_lossy());
        }
    }
}
