# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""What each release channel serves for one exact version.

Two subcommands share one probe per channel:

- ``wait --channel npm|crates|pypi|conda|scoop|copr --version X.Y.Z``: setuptest runs it
  before installing from a registry, so a channel that is still propagating
  is waited for (bounded, ``--timeout``) instead of failing the leg. Only the
  exact version counts: ``0.5.30`` is not ``0.5.3``.
- ``crate --name N --version X.Y.Z``: rustready asks it whether to
  publish: exit 0 published (skip), 10 absent (publish), 1 when crates.io
  answers 429, a 5xx or nothing after the retries (stop, never publish).
- ``report --run ID --sha SHA --tag vX.Y.Z``: runreport.yml runs it when
  a release run completes, whatever its conclusion. It writes one summary:
  the run and every job that did not succeed, whether the tag points at the
  run's commit, the pypiindex.yml run for that commit, and what the GitHub
  release, crates.io, npm, Homebrew, PyPI, the release's .deb and .rpm,
  conda, scoop, COPR, Chocolatey and winget serve. Chocolatey and winget
  are reviewed by people: a version still under review reads "in
  moderation" and does not fail the report. The summary is written first;
  the exit code then fails the report when anything is missing.

Endpoints (overridable for tests through the environment):

- npm: ``$NPM_REGISTRY/@urna%2fcli/<version>`` (the version document);
- crates.io: ``$CRATES_INDEX/ur/na/urna`` (the sparse index cargo and
  binstall resolve through), a non-yanked line with that ``vers``;
- PyPI: ``$PYPI_URL/pypi/urna/<version>/json``, with files;
- Homebrew: ``$RAW_GITHUB/hoffresearch/homebrew-urna/main/Formula/urna.rb``;
- conda: ``$ANACONDA_API/release/hoffresearch/urna/<version>``, all 5 platforms;
- scoop: ``$RAW_GITHUB/hoffresearch/scoop-urna/main/bucket/urna.json``;
- COPR: ``$COPR_API/package/`` of hoffresearch/urna, its last succeeded build;
- Chocolatey: ``$CHOCOLATEY_API/Packages(Id='urna',Version='<version>')``;
- winget: the manifest in ``microsoft/winget-pkgs`` on ``$RAW_GITHUB``, or
  an open pull request for the version (GitHub search);
- GitHub: ``$GITHUB_API_URL`` (release, run, jobs, tag ref), ``GH_TOKEN``.

    python tool/tasks/chanprobe.py wait --channel npm --version 0.5.3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

UA = "urna-release (github.com/hoffresearch/urna)"
SERVED = "served"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).rstrip("/")


def _get(url: str, token: bool = False, accept: str = "application/json") -> tuple[int, str]:
    """(status, body); status 0 when the host does not answer."""
    headers = {"User-Agent": UA, "Accept": accept}
    if token and (tok := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, ""
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return 0, str(err)


def _load(body: str):
    """The JSON in a body, or None when it is not JSON: never a guess."""
    try:
        return json.loads(body) if body else None
    except ValueError:
        return None


def _doc(status: int, body: str) -> dict:
    doc = _load(body) if status == 200 else None
    return doc if isinstance(doc, dict) else {}


def _state(status: int, ok: bool, what: str) -> str:
    if status == 200 and ok:
        return SERVED
    if status in (200, 404):
        return f"absent ({what})"
    return f"unavailable (HTTP {status or 'no answer'})"


def probe_npm(version: str) -> str:
    status, body = _get(
        f"{_env('NPM_REGISTRY', 'https://registry.npmjs.org')}/@urna%2fcli/{version}"
    )
    doc = _doc(status, body)
    return _state(status, doc.get("version") == version, f"npm has no @urna/cli {version}")


def probe_crates(version: str) -> str:
    status, body = _get(f"{_env('CRATES_INDEX', 'https://index.crates.io')}/ur/na/urna")
    lines = [_load(ln) for ln in body.splitlines()] if status == 200 else []
    ok = any(
        isinstance(ln, dict) and ln.get("vers") == version and ln.get("yanked") is False
        for ln in lines
    )
    return _state(status, ok, f"the index has no unyanked urna {version}")


def probe_pypi(version: str) -> str:
    status, body = _get(f"{_env('PYPI_URL', 'https://pypi.org')}/pypi/urna/{version}/json")
    doc = _doc(status, body)
    files = doc.get("urls") or []
    ok = doc.get("info", {}).get("version") == version and bool(files)
    state = _state(status, ok, f"pypi has no urna {version} files")
    return f"{SERVED} ({len(files)} files)" if state == SERVED else state


def probe_homebrew(version: str) -> str:
    raw = _env("RAW_GITHUB", "https://raw.githubusercontent.com")
    status, body = _get(f"{raw}/hoffresearch/homebrew-urna/main/Formula/urna.rb")
    found = re.search(r'^\s*version "([^"]+)"', body, re.M) if status == 200 else None
    at = found.group(1) if found else None
    return _state(status, at == version, f"the formula is at {at}")


def probe_github(version: str, repo: str) -> str:
    api = _env("GITHUB_API_URL", "https://api.github.com")
    status, body = _get(f"{api}/repos/{repo}/releases/tags/v{version}", token=True)
    assets = _doc(status, body).get("assets") or []
    state = _state(status, bool(assets), f"no release v{version}")
    return f"{SERVED} ({len(assets)} assets)" if state == SERVED else state


CONDA_PLATFORMS = 5
PACKAGES = ("{v}-1_amd64.deb", "{v}-1_arm64.deb", "{v}-1.x86_64.rpm", "{v}-1.aarch64.rpm")
MODERATED = "in moderation"
PUBLISHED_STATUS = ("Approved", "Exempted")


def probe_conda(version: str) -> str:
    api = _env("ANACONDA_API", "https://api.anaconda.org")
    status, body = _get(f"{api}/release/hoffresearch/urna/{version}")
    files = _doc(status, body).get("distributions") or []
    state = _state(status, len(files) == CONDA_PLATFORMS, f"conda has {len(files)} of 5 platforms")
    return f"{SERVED} ({len(files)} platforms)" if state == SERVED else state


def probe_scoop(version: str) -> str:
    raw = _env("RAW_GITHUB", "https://raw.githubusercontent.com")
    status, body = _get(f"{raw}/hoffresearch/scoop-urna/main/bucket/urna.json")
    at = _doc(status, body).get("version")
    return _state(status, at == version, f"the bucket is at {at}")


def probe_copr(version: str) -> str:
    api = _env("COPR_API", "https://copr.fedorainfracloud.org/api_3")
    query = "ownername=hoffresearch&projectname=urna&packagename=urna"
    status, body = _get(f"{api}/package/?{query}&with_latest_succeeded_build=true")
    build = (_doc(status, body).get("builds") or {}).get("latest_succeeded") or {}
    at = (build.get("source_package") or {}).get("version") or ""
    return _state(status, at.startswith(f"{version}-"), f"copr last built {at or 'nothing'}")


def probe_packages(version: str, repo: str) -> str:
    api = _env("GITHUB_API_URL", "https://api.github.com")
    status, body = _get(f"{api}/repos/{repo}/releases/tags/v{version}", token=True)
    names = {a.get("name") for a in _doc(status, body).get("assets") or []}
    want = [f"urna{'_' if n.endswith('.deb') else '-'}{n.format(v=version)}" for n in PACKAGES]
    lack = [n for n in want if n not in names]
    return _state(status, not lack, f"the release lacks {', '.join(lack)}")


def probe_chocolatey(version: str) -> str:
    """Served once approved or exempted from moderation; a package the
    moderators still review is in moderation, which is not a failure of the
    release."""
    api = _env("CHOCOLATEY_API", "https://community.chocolatey.org/api/v2")
    url = f"{api}/Packages(Id='urna',Version='{version}')"
    status, body = _get(url, accept="application/atom+xml")
    found = re.search(r"<d:PackageStatus>([^<]*)</d:PackageStatus>", body)
    at = found.group(1) if found and status == 200 else None
    if at and at not in PUBLISHED_STATUS:
        return f"{MODERATED} ({at})"
    return _state(status, at in PUBLISHED_STATUS, f"chocolatey has no urna {version}")


def probe_winget(version: str) -> str:
    """Served once the manifest is in winget-pkgs; an open pull request for
    the version is in moderation."""
    raw = _env("RAW_GITHUB", "https://raw.githubusercontent.com")
    path = f"microsoft/winget-pkgs/master/manifests/h/HoffResearch/Urna/{version}"
    status, _ = _get(f"{raw}/{path}/HoffResearch.Urna.yaml")
    if status == 200:
        return SERVED
    api = _env("GITHUB_API_URL", "https://api.github.com")
    q = f"repo:microsoft/winget-pkgs+is:pr+is:open+in:title+HoffResearch.Urna+{version}"
    found_status, body = _get(f"{api}/search/issues?q={q}", token=True)
    items = _doc(found_status, body).get("items") or []
    pr = next((i for i in items if i.get("title", "").endswith(f" {version}")), None)
    if pr:
        return f"{MODERATED} (#{pr.get('number')})"
    return _state(status, False, f"winget-pkgs has no HoffResearch.Urna {version}")


PROBES = {
    "npm": probe_npm,
    "crates": probe_crates,
    "pypi": probe_pypi,
    "conda": probe_conda,
    "scoop": probe_scoop,
    "copr": probe_copr,
}

PUBLISHED, ABSENT = 0, 10


def crate_status(name: str, version: str, attempts: int = 5, delay: float = 5) -> tuple[int, str]:
    """(PUBLISHED | ABSENT | 1, why) for one crate version on crates.io.

    Only a 200 means published and only a 404 means absent. A 429, a 5xx or
    no answer is retried, then reported as an error: rustready must not
    read an outage as "not published yet" and try to publish over it.
    """
    api = _env("CRATES_API", "https://crates.io/api/v1")
    for attempt in range(1, attempts + 1):
        status, _ = _get(f"{api}/crates/{name}/{version}")
        if status == 200:
            return PUBLISHED, f"{name} {version} is on crates.io"
        if status == 404:
            return ABSENT, f"{name} {version} is not on crates.io"
        why = f"crates.io answered {status or 'nothing'} for {name} {version}"
        print(f"chanprobe: {why} (attempt {attempt} of {attempts})", flush=True)
        if attempt < attempts:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return 1, why


def wait(channel: str, version: str, timeout: float, interval: float) -> tuple[bool, str]:
    deadline, delay = time.monotonic() + timeout, interval
    while True:
        state = PROBES[channel](version)
        print(f"chanprobe: {channel} {version}: {state}", flush=True)
        if state.startswith(SERVED):
            return True, state
        if time.monotonic() + delay > deadline:
            return False, f"{channel} did not serve {version} within {timeout:.0f}s: {state}"
        time.sleep(delay)
        delay = min(delay * 2, 60)


def _json(url: str, token: bool = False) -> tuple[int, dict]:
    status, body = _get(url, token)
    return status, _doc(status, body)


def _unavailable(status: int) -> str:
    return f"unavailable (HTTP {status or 'no answer'})"


def wheels_and_attestations(version: str, repo: str) -> tuple[list[str], list[str]]:
    """(summary lines, problems): the release's wheels against PyPI's, and an
    attestation for every archive and wheel on the release (by digest)."""
    api = _env("GITHUB_API_URL", "https://api.github.com")
    lines, problems = [], []
    status, release = _json(f"{api}/repos/{repo}/releases/tags/v{version}", token=True)
    if status != 200:
        return lines, [f"the GitHub release v{version} is {_unavailable(status)}"]
    assets = release.get("assets") or []
    digest = {a["name"]: str(a.get("digest") or "").removeprefix("sha256:") for a in assets}
    status, pypi = _json(f"{_env('PYPI_URL', 'https://pypi.org')}/pypi/urna/{version}/json")
    on_pypi = {f["filename"]: f["digests"]["sha256"] for f in pypi.get("urls") or []}
    on_release = {n: d for n, d in digest.items() if n.endswith(".whl")}
    if status not in (200, 404):
        problems.append(f"PyPI's urna {version} files are {_unavailable(status)}")
    elif not on_release:
        problems.append("the GitHub release carries no wheels")
    elif on_release != on_pypi:
        differ = sorted(
            n for n in set(on_release) | set(on_pypi) if on_release.get(n) != on_pypi.get(n)
        )
        problems.append(f"PyPI and the GitHub release differ on {differ}")
    else:
        lines.append(f"Wheels: PyPI and the GitHub release carry the same {len(on_release)} files")
    signed = sorted(n for n in digest if n.endswith((".tar.xz", ".zip", ".whl")))
    missing, unknown = [], []
    for name in signed:
        if not digest[name]:
            missing.append(name)
            continue
        status, found = _json(f"{api}/repos/{repo}/attestations/sha256:{digest[name]}", True)
        if status not in (200, 404):
            unknown.append(name)
        elif not found.get("attestations"):
            missing.append(name)
    if not signed:
        problems.append("the GitHub release carries no archives or wheels")
    if missing:
        problems.append(f"no attestation for {missing}")
    if unknown:
        problems.append(f"the attestations of {unknown} are unavailable")
    if signed and not missing and not unknown:
        lines.append(
            f"Attestations: every archive and wheel on the release has one ({len(signed)})"
        )
    return lines, problems


def _tag_commit(api: str, repo: str, tag: str) -> str | None:
    obj = _json(f"{api}/repos/{repo}/git/ref/tags/{tag}", True)[1].get("object") or {}
    if obj.get("type") == "tag":  # annotated: one more hop to the commit
        obj = _json(f"{api}/repos/{repo}/git/tags/{obj.get('sha')}", True)[1].get("object") or {}
    return obj.get("sha")


def report(run_id: str, sha: str, tag: str, repo: str) -> tuple[bool, str]:
    """The markdown summary and whether everything was released."""
    api = _env("GITHUB_API_URL", "https://api.github.com")
    version = tag.removeprefix("v")
    problems: list[str] = []
    status, run = _json(f"{api}/repos/{repo}/actions/runs/{run_id}", True)
    conclusion = run.get("conclusion") or (
        run.get("status") if status == 200 else _unavailable(status)
    )
    if conclusion != "success":
        problems.append(f"the release run ended {conclusion}")
    lines = [f"## Release {tag}", "", f"Release run {run_id}: **{conclusion}** at `{sha[:12]}`", ""]
    commit = _tag_commit(api, repo, tag)
    if commit != sha:
        problems.append(f"tag {tag} points at {commit}, not the run's {sha[:12]}")
    lines.append(
        f"Tag {tag} -> `{(commit or 'missing')[:12]}`"
        + ("" if commit == sha else " (not this run)")
    )
    status, found = _json(f"{api}/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", True)
    jobs = found.get("jobs") or []
    off = [j for j in jobs if j.get("conclusion") != "success"]
    if status != 200 or not jobs:
        problems.append(
            f"the run's jobs are {_unavailable(status) if status != 200 else 'missing'}"
        )
    lines += ["", f"Jobs: {len(jobs) - len(off)} of {len(jobs)} succeeded"]
    lines += [f"- {j.get('name')}: {j.get('conclusion') or j.get('status')}" for j in off]
    url = f"{api}/repos/{repo}/actions/workflows/pypiindex.yml/runs?head_sha={sha}"
    status, found = _json(url, True)
    pypi_runs = found.get("workflow_runs") or []
    if status != 200:
        pypi_state = _unavailable(status)
    elif not pypi_runs:
        pypi_state = "no run for this commit"
    else:
        pypi_state = pypi_runs[0].get("conclusion") or pypi_runs[0].get("status") or "unknown"
    if pypi_state != "success":
        problems.append(f"pypiindex.yml: {pypi_state}")
    lines.append(f"pypiindex.yml: {pypi_state}")
    channels = {
        "GitHub release": probe_github(version, repo),
        "crates.io": probe_crates(version),
        "npm": probe_npm(version),
        "Homebrew": probe_homebrew(version),
        "PyPI": probe_pypi(version),
        ".deb and .rpm": probe_packages(version, repo),
        "conda": probe_conda(version),
        "scoop": probe_scoop(version),
        "COPR": probe_copr(version),
        "Chocolatey": probe_chocolatey(version),
        "winget": probe_winget(version),
    }
    lines += ["", "| channel | " + version + " |", "|---|---|"]
    lines += [f"| {name} | {state} |" for name, state in channels.items()]
    problems += [
        f"{name}: {state}"
        for name, state in channels.items()
        if not state.startswith((SERVED, MODERATED))
    ]
    extra, issues = wheels_and_attestations(version, repo)
    lines += ["", *extra]
    problems += issues
    lines += ["", "Released everywhere." if not problems else "Not released everywhere:"]
    lines += [f"- {p}" for p in problems]
    return not problems, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait")
    w.add_argument("--channel", choices=sorted(PROBES), required=True)
    w.add_argument("--version", required=True)
    w.add_argument("--timeout", type=float, default=900)
    w.add_argument("--interval", type=float, default=10)
    c = sub.add_parser("crate")
    c.add_argument("--name", required=True)
    c.add_argument("--version", required=True)
    c.add_argument("--attempts", type=int, default=5)
    c.add_argument("--delay", type=float, default=5)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--sha", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--repo", default="hoffresearch/urna")
    args = parser.parse_args(argv)
    if args.cmd == "wait":
        ok, why = wait(args.channel, args.version.removeprefix("v"), args.timeout, args.interval)
        print(f"chanprobe: {why}", file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    if args.cmd == "crate":
        code, why = crate_status(args.name, args.version, args.attempts, args.delay)
        print(f"chanprobe: {why}", file=sys.stderr if code == 1 else sys.stdout)
        return code
    try:
        ok, text = report(args.run, args.sha, args.tag, args.repo)
    except Exception as err:  # noqa: BLE001 (the summary must survive any answer)
        ok = False
        text = f"## Release {args.tag}\n\nRelease run {args.run} at `{args.sha[:12]}`: "
        text += f"the report could not finish ({type(err).__name__}: {err}).\n"
    print(text)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
