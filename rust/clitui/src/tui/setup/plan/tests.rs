//! The plan's cases: which steps run, blocked or ticked, from a scan
//! and the flags, with the catalogue's uv and python range.

use super::*;
use crate::tui::setup::scan::{Channel, Payload};
use std::path::PathBuf;

fn bare() -> Scan {
    Scan {
        version: "0.5.0",
        name: "0.5.0",
        target: "aarch64-macos".into(),
        exe: PathBuf::from("/opt/homebrew/bin/urna"),
        channel: Channel::Homebrew,
        home: Some(PathBuf::from("/h/.local/share/urna")),
        embedder: None,
        payload: None,
        python: Some(("python3".into(), "Python 3.12.4".into())),
        pinned_python: None,
        deps: false,
        venv: false,
        venv_version: None,
        base_python: Some("python3".into()),
        uv: None,
        curl: Some(PathBuf::from("/usr/bin/curl")),
        simd: "neon",
        kit: None,
    }
}

#[test]
fn a_fresh_brew_install_plans_everything() {
    let p = plan(&bare(), &Opts::default());
    assert!(p.iter().filter(|i| i.task != Task::Models).all(Item::runs));
    assert!(!p[2].runs(), "models are opt-in");
    // no uv on the machine: the payload's catalogue names the one to fetch.
    assert!(
        p[1].detail
            .contains("via uv (downloaded after the payload, sha256 checked)"),
        "{}",
        p[1].detail
    );
    let mut no_curl = bare();
    no_curl.curl = None;
    assert!(
        plan(&no_curl, &Opts::default())[1]
            .detail
            .contains("via pip")
    );
}

/// `bare` with a payload whose catalogue names uv and the python range.
fn with_upstreams() -> Scan {
    let mut s = installed(Some("0.5.0"));
    let mut catalog = models::tests::sample();
    let mut archives = std::collections::BTreeMap::new();
    archives.insert(
        "aarch64-macos".to_string(),
        models::Archive {
            url: "https://example/uv.tar.gz".into(),
            sha256: "ab".repeat(32),
        },
    );
    catalog.uv = Some(models::UvRelease {
        version: "0.13.0".into(),
        archives,
    });
    catalog.python = Some(models::PythonRange {
        requires: ">=3.10,<3.15".into(),
        torch: "2.14.1".into(),
    });
    s.kit = Some(models::Kit {
        script: PathBuf::from("/h/.local/share/urna/python/urna/model/installer.py"),
        catalog,
    });
    s
}

#[test]
fn the_env_is_planned_on_the_catalogue_uv_and_range() {
    let mut s = with_upstreams();
    s.deps = false;
    let p = plan(&s, &Opts::default());
    assert!(p[1].runs());
    assert!(
        p[1].detail
            .contains("via uv 0.13.0 (downloaded, sha256 checked) on python 3.10 to 3.14"),
        "{}",
        p[1].detail
    );
    // a platform the catalogue has no uv for falls back to pip.
    s.target = "riscv64-linux".into();
    assert!(
        plan(&s, &Opts::default())[1]
            .detail
            .contains("via pip on python 3.10 to 3.14")
    );
}

#[test]
fn an_env_on_a_python_outside_the_range_is_rebuilt() {
    let mut s = with_upstreams();
    s.venv = true;
    s.venv_version = Some((3, 15));
    let p = plan(&s, &Opts::default());
    assert!(p[1].runs(), "a ready env on 3.15 still has to move");
    assert!(
        p[1].detail.contains(
            "runs python 3.15, outside python 3.10 to 3.14: it is deleted and built again"
        ),
        "{}",
        p[1].detail
    );
    assert!(p[1].detail.contains("cached get their packages back"));
    s.venv_version = Some((3, 14));
    assert!(
        !plan(&s, &Opts::default())[1].runs(),
        "inside the range it stays"
    );
    s.venv_version = Some((3, 15));
    s.pinned_python = Some("/opt/py/bin/python".into());
    assert!(
        !plan(&s, &Opts::default())[1].detail.contains("deleted"),
        "URNA_PYTHON wins: setup does not touch an env queries do not run"
    );
}

#[test]
fn a_ready_machine_only_verifies_unless_forced() {
    let mut s = bare();
    s.embedder = Some(PathBuf::from(
        "/h/.local/share/urna/python/urna/embed/potionqry.py",
    ));
    s.payload = Some(Payload {
        version: Some("0.5.0".into()),
        missing: Vec::new(),
    });
    s.deps = true;
    let p = plan(&s, &Opts::default());
    assert_eq!(p.iter().filter(|i| i.runs()).count(), 1);
    let forced = plan(
        &s,
        &Opts {
            force: true,
            ..Opts::default()
        },
    );
    assert!(forced[0].runs());
}

fn installed(version: Option<&str>) -> Scan {
    let mut s = bare();
    s.embedder = Some(PathBuf::from(
        "/h/.local/share/urna/python/urna/embed/potionqry.py",
    ));
    s.payload = Some(Payload {
        version: version.map(String::from),
        missing: Vec::new(),
    });
    s.deps = true;
    s
}

#[test]
fn an_upgraded_binary_replaces_the_payload_of_another_release() {
    // the binary is 0.5.0 (bare); a 0.4.9 payload and an unstamped one
    // (0.5.1's, from before the stamp) are both replaced, the venv step
    // stays off.
    for have in [Some("0.4.9"), None] {
        let p = plan(&installed(have), &Opts::default());
        assert!(p[0].runs(), "{have:?}");
        assert!(p[0].detail.contains("v0.5.0 is wanted"), "{}", p[0].detail);
        assert!(!p[1].runs());
    }
    let unstamped = plan(&installed(None), &Opts::default());
    assert!(unstamped[0].detail.contains("unstamped"));
}

#[test]
fn a_payload_missing_a_required_file_is_repaired() {
    // stamped with the wanted release, but a file is gone: the payload
    // step runs and names the file; the venv step stays off.
    let mut s = installed(Some("0.5.0"));
    if let Some(p) = s.payload.as_mut() {
        p.missing = vec!["searchtxt.py".into()];
    }
    let p = plan(&s, &Opts::default());
    assert!(p[0].runs(), "{}", p[0].detail);
    assert!(
        p[0].detail.contains("missing searchtxt.py"),
        "{}",
        p[0].detail
    );
    assert!(!p[1].runs());
    let skip = Opts {
        no_payload: true,
        ..Opts::default()
    };
    assert!(!plan(&s, &skip)[0].runs());
}

#[test]
fn a_payload_in_another_data_root_does_not_count_as_installed() {
    // URNA_DATA_DIR points setup at an empty dir while an older install
    // sits in ~/.local/share: setup installs into its own dir.
    let mut s = bare();
    s.embedder = Some(PathBuf::from(
        "/h/.local/share/urna/python/urna/embed/potionqry.py",
    ));
    s.home = Some(PathBuf::from("/tmp/data/urna"));
    s.deps = true;
    let p = plan(&s, &Opts::default());
    assert!(p[0].runs(), "{}", p[0].detail);
    assert!(
        p[0].detail.contains("resolves until then"),
        "{}",
        p[0].detail
    );
}

#[test]
fn version_picks_the_payload_release_and_matching_skips_it() {
    let pinned = Opts {
        version: Some("v0.4.9".into()),
        ..Opts::default()
    };
    assert_eq!(wanted_version(&bare(), &pinned), "0.4.9");
    assert!(!plan(&installed(Some("0.4.9")), &pinned)[0].runs());
    assert!(plan(&installed(Some("0.5.0")), &pinned)[0].runs());
    let fresh = plan(&bare(), &pinned);
    assert!(
        fresh[0].detail.contains("v0.4.9 release"),
        "{}",
        fresh[0].detail
    );
    let skip = Opts {
        no_payload: true,
        ..Opts::default()
    };
    assert!(!plan(&installed(Some("0.4.9")), &skip)[0].runs());
}

#[test]
fn a_pinned_interpreter_without_deps_blocks_the_python_step() {
    // URNA_PYTHON wins over any venv setup would build, so a pinned
    // interpreter lacking the deps is a block with the variable named,
    // and one that has them needs no python step at all. the scan
    // carries the pin; the plan never reads the process env, so this
    // holds whatever the gate exports.
    let mut s = bare();
    s.pinned_python = Some("/opt/py/bin/python".into());
    let p = plan(&s, &Opts::default());
    assert!(p[1].blocked.as_deref().unwrap().contains("URNA_PYTHON"));
    assert!(p[1].detail.contains("/opt/py/bin/python"));
    assert_eq!(blocked_code(&p), 14);
    s.deps = true;
    let p = plan(&s, &Opts::default());
    assert!(p[1].blocked.is_none());
    assert!(!p[1].runs());
    assert_eq!(blocked_code(&p), 0);
}

#[test]
fn missing_tools_block_with_a_reason() {
    let mut s = bare();
    s.curl = None;
    s.base_python = None;
    let p = plan(&s, &Opts::default());
    assert!(p[0].blocked.as_deref().unwrap().contains("curl"));
    assert_eq!(blocked_code(&p), 14);
    assert_eq!(blocked_code(&plan(&bare(), &Opts::default())), 0);
    assert!(!p[1].runs());
    assert!(p[3].runs() && p[3].task == Task::Verify);
}

fn with_models(models: &[&str], allow: &[&str]) -> (Scan, Opts) {
    let mut s = bare();
    s.kit = Some(models::Kit {
        script: PathBuf::from("/h/.local/share/urna/python/urna/model/installer.py"),
        catalog: models::tests::sample(),
    });
    let o = Opts {
        models: models.iter().map(|m| m.to_string()).collect(),
        allow_remote_code: allow.iter().map(|m| m.to_string()).collect(),
        ..Opts::default()
    };
    (s, o)
}

#[test]
fn models_are_opt_in_named_by_the_catalog_and_sized() {
    let (s, o) = with_models(&[], &[]);
    let item = &plan(&s, &o)[2];
    assert!(
        !item.runs() && item.detail.contains("2 offered"),
        "{}",
        item.detail
    );
    let (s, o) = with_models(&["org/a"], &[]);
    let item = &plan(&s, &o)[2];
    assert!(item.runs(), "{:?}", item.blocked);
    assert!(item.detail.contains("a: p + ") && item.detail.contains("huggingface.co (org/a@aaaa)"));
    let (s, o) = with_models(&["all"], &["b"]);
    let item = &plan(&s, &o)[2];
    assert!(item.runs() && item.detail.contains("b: ") && item.detail.contains("a: "));
}

#[test]
fn an_excluded_model_or_unallowed_repo_code_blocks_with_the_reason() {
    let (s, o) = with_models(&["big"], &[]);
    let item = &plan(&s, &o)[2];
    let why = item.blocked.clone().unwrap();
    assert!(
        why.contains("not offered") && why.contains("too heavy"),
        "{why}"
    );
    assert_eq!(
        blocked_code(&plan(&s, &o)),
        super::super::job::codes::BLOCKED
    );
    let (s, o) = with_models(&["b"], &[]);
    let why = plan(&s, &o)[2].blocked.clone().unwrap();
    assert!(why.contains("--allow-remote-code b"), "{why}");
    let (s, o) = with_models(&["b"], &["a"]);
    assert!(plan(&s, &o)[2].blocked.is_some(), "consent is per model");
}

#[test]
fn models_go_only_into_the_managed_venv_and_wait_for_the_payload() {
    let (mut s, o) = with_models(&["a"], &[]);
    s.pinned_python = Some("/usr/bin/python3".into());
    let why = plan(&s, &o)[2].blocked.clone().unwrap();
    assert!(why.contains("URNA_PYTHON pins /usr/bin/python3"), "{why}");
    s.pinned_python = Some("/h/.local/share/urna/venv/bin/python".into());
    assert!(plan(&s, &o)[2].blocked.is_none() || cfg!(windows));
    let (mut s, o) = with_models(&["a"], &[]);
    s.kit = None;
    let item = &plan(&s, &o)[2];
    assert!(
        item.runs() && item.detail == "after the payload: a",
        "{}",
        item.detail
    );
}

#[test]
fn chosen_models_build_the_managed_venv_even_when_python_has_the_deps() {
    let (mut s, o) = with_models(&["a"], &[]);
    s.deps = true;
    s.uv = Some(PathBuf::from("/u/uv"));
    let p = plan(&s, &o);
    assert!(p[1].runs() && p[1].detail.contains("models' packages go only there"));
    s.venv = true;
    assert!(!plan(&s, &o)[1].runs(), "an existing managed venv is kept");
    let (mut s, _) = with_models(&[], &[]);
    s.deps = true;
    assert!(!plan(&s, &Opts::default())[1].runs(), "no models: no venv");
}

#[test]
fn picking_a_model_in_the_picker_turns_the_managed_venv_on() {
    // an external python with the base deps and no managed venv: the plan
    // starts with the python step off; choosing a model must turn it on,
    // and dropping the model turns it off again. the payload tick the
    // user gave stays.
    let (mut s, mut o) = with_models(&[], &[]);
    s.deps = true;
    s.uv = Some(PathBuf::from("/u/uv"));
    let mut items = plan(&s, &o);
    assert!(!items[1].runs() && !items[2].runs());
    items[0].on = true;
    o.models = vec!["a".into()];
    reselect(&mut items, &s, &o);
    assert!(items[1].runs(), "{}", items[1].detail);
    assert!(items[1].detail.contains("models' packages go only there"));
    assert!(items[2].runs() && items[0].on);
    o.models.clear();
    reselect(&mut items, &s, &o);
    assert!(!items[1].runs() && !items[2].runs() && items[0].on);
}
