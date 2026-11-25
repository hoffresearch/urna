//! The needs contract of common/needs.rs, which every test that can lack an
//! external resource includes: a need the run does not require logs one
//! skip line, one it requires fails, and a need the table lacks is refused.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code: a failing unwrap is a failing test"
)]

// `lack` reads the process environment, where a skip would land in the
// gate's own skip log; its body is `decide`, which these tests drive.
#[allow(dead_code, reason = "lack is decide over the process environment")]
#[path = "common/needs.rs"]
mod needs;

use std::collections::HashMap;
use std::ffi::OsString;

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> + use<> {
    let map: HashMap<String, OsString> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), OsString::from(v)))
        .collect();
    move |k| map.get(k).cloned()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("urna_needs_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn an_optional_need_appends_one_line_to_the_skip_log() {
    let dir = scratch("optional");
    let log = dir.join("skips.log");
    let path = log.to_str().unwrap();
    let with_log = env(&[("URNA_SKIP_LOG", path)]);
    needs::decide("cards", "t1", "URNA_CARDS_CORPUS unset", &with_log);
    needs::decide("lfs", "t2", "a pointer", &with_log);
    // another need's variable, or a value other than 1, requires nothing.
    let other = env(&[
        ("URNA_REQUIRE_NATIVE", "1"),
        ("URNA_REQUIRE_FORGE", "0"),
        ("URNA_SKIP_LOG", path),
    ]);
    needs::decide("forge", "t3", "no numpy", &other);
    assert_eq!(
        std::fs::read_to_string(&log).unwrap(),
        "cards\tt1\tURNA_CARDS_CORPUS unset\nlfs\tt2\ta pointer\nforge\tt3\tno numpy\n"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[should_panic(
    expected = "forge is required (URNA_REQUIRE_FORGE or URNA_REQUIRE_ALL): t: no numpy; provide a python with numpy and tokenizers"
)]
fn the_need_s_own_variable_turns_the_skip_into_a_failure() {
    needs::decide(
        "forge",
        "t",
        "no numpy",
        &env(&[("URNA_REQUIRE_FORGE", "1")]),
    );
}

#[test]
#[should_panic(expected = "cards is required (URNA_REQUIRE_CARDS or URNA_REQUIRE_ALL)")]
fn urna_require_all_requires_every_need() {
    needs::decide("cards", "t", "unset", &env(&[("URNA_REQUIRE_ALL", "1")]));
}

#[test]
fn a_required_need_logs_nothing() {
    let dir = scratch("required");
    let log = dir.join("skips.log");
    let vars = env(&[
        ("URNA_REQUIRE_ALL", "1"),
        ("URNA_SKIP_LOG", log.to_str().unwrap()),
    ]);
    let failed = std::panic::catch_unwind(|| needs::decide("lfs", "t", "a pointer", &vars));
    assert!(failed.is_err(), "a required need passed");
    assert!(!log.exists(), "a failure is not a skip: nothing logged");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[should_panic(expected = "forgee: not a need of tool/tests/testneeds.tsv")]
fn a_need_the_table_lacks_is_refused() {
    needs::decide("forgee", "t", "a typo", &env(&[]));
}
