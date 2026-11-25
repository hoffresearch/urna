"""every check CI runs has its local step, or the reason it has none.

tool/tasks/gatesteps.tsv maps each step of the gate workflows to the tier
of tool/tasks/fullcheck.sh that runs the same command. this reads the
workflows (pyyaml), the map and the tiers' source, and fails when:

- a step of gatecheck.yml or lintworks.yml has no line in the map (the
  other workflows it lists, rehearsal.yml and cleanhost.yml, are partial);
- a line names a workflow, job or step that no longer exists;
- a line's local step is not a step of a tier it names;
- a command of an `exact` line is not a command of the function that step
  runs, in every tier the line names: the line itself, which may carry
  environment assignments before it and a redirection after it, and
  nothing else. a `step "..."` label, a comment or the same command in
  another step or tier does not count;
- a value is outside the map's vocabulary, or a `none` line gives no
  reason and mitigation.

the cases after the tree's own prove the red side on copies: a CI step
dropped, a CI step added, a local step dropped, a flag changed on either
side, a command left only in another tier, a command left only in its
label.

    python tool/tests/test_gatematch.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(
        "test_gatematch: needs pyyaml, in pyproject.toml's gate group: "
        "uv sync --group forge --group gate (pip install --group gate)"
    )

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
MAP = REPO / "tool" / "tasks" / "gatesteps.tsv"
FULLCHECK = REPO / "tool" / "tasks" / "fullcheck.sh"
TIERSTEPS = REPO / "tool" / "tasks" / "tiersteps.sh"

COMPLETE = ("gatecheck.yml", "lintworks.yml")
PARTIAL = ("rehearsal.yml", "cleanhost.yml")
# actions that set a job up and check nothing.
INFRA = (
    "actions/checkout",
    "actions/setup-python",
    "Swatinem/rust-cache",
    "actions/cache",
    "actions/upload-artifact",
    "actions/download-artifact",
)
TIERS = ("quick", "push", "full")
MATCHES = ("exact", "equiv", "setup", "none")
# steps of every tier that are not `cached` calls, each with the text in
# fullcheck.sh that is it.
BUILTIN = {"report": "testneeds.py report", "pinmode": "PIN_HNSW_DIGESTS"}
FULL_MARK = "# ---- tier full: the release gate on the working tree ----"


def label(step: dict) -> str:
    uses = str(step.get("uses", "")).split("@")[0]
    first = str(step.get("run", "")).strip().splitlines()[:1]
    return step.get("name") or step.get("id") or uses or (first[0] if first else "")


def ci_steps(texts: dict[str, str]) -> dict[tuple[str, str, str], dict]:
    """(workflow, job, step label) -> the step, minus the setup actions."""
    out = {}
    for wf, text in texts.items():
        for job, body in (yaml.safe_load(text).get("jobs") or {}).items():
            for step in body.get("steps") or []:
                if str(step.get("uses", "")).split("@")[0] in INFRA:
                    continue
                out[(wf, job, label(step))] = step
    return out


def functions(text: str) -> dict[str, str]:
    """shell function -> its body, for the `name() {` ... `}` definitions."""
    return dict(re.findall(r"^([a-z_0-9]+)\(\) \{\n(.*?)^\}", text, re.M | re.S))


def step_calls(text: str) -> dict[str, set[str]]:
    """step name -> the functions it runs, from `cached NAME ... -- FUNCTION`
    and `uncached NAME -- FUNCTION` (a `cached` call may continue a line)."""
    out: dict[str, set[str]] = {}
    call = r"^\s*(?:cached|uncached)\s+([a-z0-9-]+)\b(?:\\\n|[^\n])*?--\s+([a-z_0-9]+)\s*$"
    for name, fn in re.findall(call, text, re.M):
        out.setdefault(name, set()).add(fn)
    return out


def tier_steps(fullcheck: str, tiersteps: str) -> dict[str, dict[str, set[str]]]:
    """tier -> its steps, each with the functions it runs. a full run is
    the push tier and then fullcheck.sh's own steps, so a step both name
    runs both functions there."""
    fns = functions(tiersteps)
    builtin: dict[str, set[str]] = {n: set() for n, text in BUILTIN.items() if text in fullcheck}
    own = fullcheck.split(FULL_MARK, 1)[1] if FULL_MARK in fullcheck else ""
    quick = step_calls(fns.get("tier_quick", ""))
    push = step_calls(fns.get("tier_push", ""))
    full = {name: set(called) for name, called in push.items()}
    for name, called in step_calls(own).items():
        full.setdefault(name, set()).update(called)
    return {"quick": quick | builtin, "push": push | builtin, "full": full | builtin}


def step_lines(
    fns: dict[str, str], names: set[str], seen: frozenset[str] = frozenset()
) -> list[str]:
    """the command lines of the functions a step runs, and of the functions
    those call by name: no blank line, no comment, no `step`/`ok` label."""
    out = []
    for fn in sorted(names - seen):
        for line in fns.get(fn, "").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "step ", "ok ")):
                continue
            out.append(line)
            word = line.split()[0]
            if word in fns:
                out += step_lines(fns, {word}, seen | names)
    return out


def commands(run: str) -> list[str]:
    """the command lines of a run script: no blank line, no comment, no
    expression (CI's context), no pip install (the environment) and no
    shell option (the tiers run under fullcheck.sh's `set -euo pipefail`)."""
    out = []
    for line in run.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "${{" in line or "pip install" in line:
            continue
        if re.fullmatch(r"set -[a-z]+", line):
            continue
        out.append(line)
    return out


def found(line: str, lines: list[str]) -> bool:
    """the CI command among a step's lines, where `python X` is `"$PY" X`
    (and `URNA_PYTHON=python` is `URNA_PYTHON="$PY"`). the local line is
    the command, after environment assignments (`NAME=value `) and before
    a redirection or a line continuation (` 2>&1 \\`), if it has them: a
    local flag CI does not pass is a difference, as CI's is."""
    wanted = {line, re.sub(r"(^|=)python3?(?= )", r'\1"$PY"', line)}
    for have in lines:
        for want in wanted:
            if have == want:
                return True
            have = re.sub(r"(?: 2>&1)?(?: \\)?$", "", have)
            if have == want:
                return True
            if have.endswith(" " + want) and re.match(r"[A-Za-z_][A-Za-z0-9_]*=", have):
                return True
    return False


def parse_map(text: str) -> list[tuple[int, list[str]]]:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line and not line.startswith("#"):
            rows.append((n, (line.split("\t") + [""] * 7)[:7]))
    return rows


def check(workflows: dict[str, str], table: str, fullcheck: str, tiersteps: str) -> list[str]:
    problems = []
    steps = ci_steps(workflows)
    tiers = tier_steps(fullcheck, tiersteps)
    fns = {**functions(tiersteps), **functions(fullcheck)}
    seen = set()
    for n, (wf, job, name, tier_col, local, match, note) in parse_map(table):
        where = f"gatesteps.tsv:{n} ({wf} {job} '{name}')"
        key = (wf, job, name)
        if key in seen:
            problems.append(f"{where}: listed twice")
        seen.add(key)
        if wf not in COMPLETE + PARTIAL:
            problems.append(f"{where}: not a workflow the map covers")
            continue
        if key not in steps:
            problems.append(f"{where}: names a CI step that no longer exists")
            continue
        if match not in MATCHES:
            problems.append(f"{where}: match {match!r} is not one of {MATCHES}")
        named = [] if tier_col == "-" else tier_col.split()
        bad = [t for t in named if t not in TIERS]
        if bad:
            problems.append(f"{where}: {bad} are not tiers")
        if match in ("exact", "equiv") and (not named or local == "-"):
            problems.append(f"{where}: an {match} line names its tiers and local step")
        if match in ("setup", "equiv") and not note:
            problems.append(f"{where}: a {match} line says how in its note")
        if match == "none" and not ("reason:" in note and "mitigation:" in note):
            problems.append(f"{where}: a none line gives its reason: and mitigation:")
        if local != "-":
            for t in named:
                if t in tiers and local not in tiers[t]:
                    problems.append(f"{where}: {local} is not a step of the {t} tier")
        if match == "exact":
            run = steps[key].get("run")
            if run is None:
                problems.append(f"{where}: an exact line needs a run step")
                continue
            for t in named:
                lines = step_lines(fns, tiers.get(t, {}).get(local, set()))
                missing = [c for c in commands(run) if not found(c, lines)]
                if missing:
                    problems.append(f"{where}: not a command of {local} in the {t} tier: {missing}")
    for key in sorted(steps):
        if key[0] in COMPLETE and key not in seen:
            wf, job, name = key
            problems.append(f"{wf} {job} '{name}': a CI step with no line in gatesteps.tsv")
    return problems


def tree() -> tuple[dict[str, str], str, str, str]:
    texts = {wf: (WORKFLOWS / wf).read_text(encoding="utf-8") for wf in COMPLETE + PARTIAL}
    return (
        texts,
        MAP.read_text(encoding="utf-8"),
        FULLCHECK.read_text(encoding="utf-8"),
        TIERSTEPS.read_text(encoding="utf-8"),
    )


def edit_job(text: str, job: str, fn) -> str:
    data = yaml.safe_load(text)
    data["jobs"][job]["steps"] = fn(data["jobs"][job]["steps"])
    return yaml.safe_dump(data, sort_keys=False)


def test_the_tree_matches() -> None:
    problems = check(*tree())
    assert not problems, "\n".join(problems)
    rows = len(parse_map(tree()[1]))
    print(f"the tree: {rows} CI steps, each with its local step or its reason: OK")


def test_a_ci_step_dropped_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    gone = "benches compile (numbers are not a gate on shared runners)"
    texts["gatecheck.yml"] = edit_job(
        texts["gatecheck.yml"], "rust", lambda s: [x for x in s if x.get("name") != gone]
    )
    problems = check(texts, table, fullcheck, tiersteps)
    assert any(gone in p and "no longer exists" in p for p in problems), problems
    print("red (a CI step removed from a copy of gatecheck.yml): OK")


def test_a_ci_step_added_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    new = {"name": "a check nobody mapped", "run": "cargo doc --workspace"}
    texts["lintworks.yml"] = edit_job(texts["lintworks.yml"], "zizmor", lambda s: [*s, new])
    problems = check(texts, table, fullcheck, tiersteps)
    assert any("a check nobody mapped" in p and "no line" in p for p in problems), problems
    print("red (a CI step with no line in the map): OK")


def test_a_local_step_dropped_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    line = '  cached benches "$RUST_IN" -- benches\n'
    assert line in tiersteps
    problems = check(texts, table, fullcheck, tiersteps.replace(line, ""))
    assert any("benches is not a step of the push tier" in p for p in problems), problems
    assert any("benches is not a step of the full tier" in p for p in problems), problems
    print("red (a step removed from a copy of tiersteps.sh's push tier): OK")


def test_a_full_step_dropped_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    marked = fullcheck.split(FULL_MARK, 1)
    own = re.sub(r"^\s*cached format-parity .*\n", "", marked[1], flags=re.M)
    problems = check(texts, table, marked[0] + FULL_MARK + own, tiersteps)
    assert any("format-parity is not a step of the full tier" in p for p in problems), problems
    print("red (a step removed from a copy of fullcheck.sh's full tier): OK")


def test_a_changed_flag_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    run = "cargo bench -p urna-engine --no-run"
    assert run in texts["gatecheck.yml"]
    texts["gatecheck.yml"] = texts["gatecheck.yml"].replace(run, run + " --all-features")
    problems = check(texts, table, fullcheck, tiersteps)
    assert any("not a command of benches" in p and "--all-features" in p for p in problems), (
        problems
    )
    print("red (a CI command whose flags the local step does not carry): OK")


def test_a_local_flag_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    line = "  cargo bench -p urna-engine --no-run\n"
    assert line in tiersteps
    edited = tiersteps.replace(line, line.rstrip() + " --all-features\n")
    problems = check(texts, table, fullcheck, edited)
    assert any("not a command of benches in the push tier" in p for p in problems), problems
    print("red (a local command with a flag CI does not pass): OK")


def test_a_command_left_in_another_tier_fails() -> None:
    # the full tier's python_tests runs test_chanprobe.py too: its copy
    # there does not stand in for the push tier's python job.
    texts, table, fullcheck, tiersteps = tree()
    line = '  "$PY" tool/tests/test_chanprobe.py\n'
    assert tiersteps.count(line) == 1
    problems = check(texts, table, fullcheck, tiersteps.replace(line, ""))
    assert any(
        "not a command of python-job in the push tier" in p and "test_chanprobe.py" in p
        for p in problems
    ), problems
    print("red (a command dropped from the push tier and kept elsewhere): OK")


def test_a_command_left_in_its_label_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    line = "  cargo build --workspace --all-targets\n"
    assert tiersteps.count(line) == 1
    assert 'step "cargo build --workspace --all-targets"' in tiersteps
    edited = tiersteps.replace(line, "  cargo build --workspace\n")
    problems = check(texts, table, fullcheck, edited)
    assert any("not a command of ci-build in the push tier" in p for p in problems), problems
    print("red (a command changed under a label that still names it): OK")


def test_a_bare_none_fails() -> None:
    texts, table, fullcheck, tiersteps = tree()
    row = "gatecheck.yml\tmiri\turna-format under miri\t-\t-\tnone\tthe schedule only\n"
    table = re.sub(r"^gatecheck\.yml\tmiri\turna-format under miri\t.*\n", row, table, flags=re.M)
    problems = check(texts, table, fullcheck, tiersteps)
    assert any("reason: and mitigation:" in p for p in problems), problems
    print("red (a none line without its reason and mitigation): OK")


def main() -> int:
    test_the_tree_matches()
    test_a_ci_step_dropped_fails()
    test_a_ci_step_added_fails()
    test_a_local_step_dropped_fails()
    test_a_full_step_dropped_fails()
    test_a_changed_flag_fails()
    test_a_local_flag_fails()
    test_a_command_left_in_another_tier_fails()
    test_a_command_left_in_its_label_fails()
    test_a_bare_none_fails()
    return 0


if __name__ == "__main__":
    sys.exit(main())
