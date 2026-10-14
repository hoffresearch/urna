"""Check where the wheels came from and which of them PyPI still needs.

pypiindex.yml runs this before its upload, in two steps:

``source``: the run that built the wheels is the one the index allows.

- ``--index pypi``: a run of ``release.yml`` started by the tag push, on
  the tag's commit, whose ``host`` job succeeded (the release run is still
  going: it waits for this upload, so only the host can have finished);
- ``--index testpypi``: a successful run of ``rehearsal.yml`` on a
  commit that is on the protected main.

``plan``: every wheel matches its ``.sha256`` (``<hex> *<name>``), there are
``--expect`` of them, all named for ``--version``; then the index's JSON for
that version decides: a file already there with the same sha256 is skipped,
one there with another sha256 fails, the rest go to ``--out``. A rerun after
a partial upload therefore sends only what is missing. The count lands in
``$GITHUB_OUTPUT`` as ``upload=N``.

    python tool/tasks/pypiindex.py source --index pypi --run 123 --repo o/r --sha abc
    python tool/tasks/pypiindex.py plan --index pypi --dir dist --version 0.5.4 --out upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = "urna"
JSON_BASE = {"pypi": "https://pypi.org/pypi", "testpypi": "https://test.pypi.org/pypi"}
SOURCE_WORKFLOW = {"pypi": "release.yml", "testpypi": "rehearsal.yml"}


class ReleaseError(Exception):
    pass


def _get_json(url: str, token: str | None = None) -> dict | None:
    """The decoded body, or None on a 404."""
    headers = {"Accept": "application/json", "User-Agent": "urna-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise ReleaseError(f"{url} answered {err.code}") from None


def check_source(index: str, run_id: str, repo: str, sha: str, main_ref: str = "origin/main"):
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    run = _get_json(f"{api}/repos/{repo}/actions/runs/{run_id}", token)
    if run is None:
        raise ReleaseError(f"run {run_id} does not exist in {repo}")
    workflow = Path(run.get("path", "").split("@")[0]).name
    if workflow != SOURCE_WORKFLOW[index]:
        raise ReleaseError(
            f"{index} takes wheels from {SOURCE_WORKFLOW[index]}, run {run_id} is {workflow}"
        )
    if index == "pypi":
        if run.get("head_sha") != sha or run.get("event") != "push":
            raise ReleaseError(f"run {run_id} is not the tag push of {sha[:12]}")
        jobs = _get_json(f"{api}/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", token)
        host = [j for j in (jobs or {}).get("jobs", []) if j.get("name") == "host"]
        if not host or host[0].get("conclusion") != "success":
            raise ReleaseError(f"run {run_id} has no successful host job: nothing was released")
        return
    if run.get("conclusion") != "success":
        raise ReleaseError(f"rehearsal run {run_id} did not succeed ({run.get('conclusion')})")
    head = run.get("head_sha", "")
    on_main = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, main_ref], capture_output=True
    )
    if not head or on_main.returncode != 0:
        raise ReleaseError(f"rehearsal run {run_id} built {head[:12]}, which is not on {main_ref}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_wheels(directory: Path, version: str, expect: int) -> dict[str, Path]:
    wheels = {p.name: p for p in sorted(directory.glob("*.whl"))}
    if len(wheels) != expect:
        raise ReleaseError(f"{directory} has {len(wheels)} wheels, expected {expect}")
    for name, path in wheels.items():
        if not name.startswith(f"{PROJECT}-{version}-"):
            raise ReleaseError(f"{name} is not a {PROJECT} {version} wheel")
        sidecar = path.with_name(name + ".sha256")
        if not sidecar.is_file():
            raise ReleaseError(f"{name} has no {sidecar.name}")
        digest, _, listed = sidecar.read_text(encoding="utf-8").strip().partition(" ")
        if listed.lstrip("*") != name or digest != _sha256(path):
            raise ReleaseError(f"{name} does not match {sidecar.name}")
    return wheels


def plan_upload(
    index: str,
    directory: Path,
    version: str,
    out: Path,
    expect: int = 4,
    json_base: str | None = None,
) -> list[str]:
    """Copy the wheels the index still needs into `out`; returns their names."""
    wheels = local_wheels(directory, version, expect)
    released = _get_json(f"{json_base or JSON_BASE[index]}/{PROJECT}/{version}/json") or {}
    remote = {f["filename"]: f["digests"]["sha256"] for f in released.get("urls", [])}
    stray = sorted(set(remote) - set(wheels))
    if stray:
        raise ReleaseError(f"{index} {version} carries files this release did not build: {stray}")
    # every remote file is checked before anything is copied: a conflict on
    # the last wheel leaves `out` untouched, not half filled.
    clashes = [n for n, p in wheels.items() if n in remote and remote[n] != _sha256(p)]
    if clashes:
        raise ReleaseError(f"{index} already has {', '.join(clashes)} with another sha256")
    upload = []
    for name, path in wheels.items():
        if name in remote:
            print(f"pypiindex: {name} already on {index}, same sha256, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / name)
        upload.append(name)
    return upload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    src = sub.add_parser("source")
    src.add_argument("--index", choices=sorted(JSON_BASE), required=True)
    src.add_argument("--run", required=True)
    src.add_argument("--repo", required=True)
    src.add_argument("--sha", required=True)
    src.add_argument("--main-ref", default="origin/main")
    plan = sub.add_parser("plan")
    plan.add_argument("--index", choices=sorted(JSON_BASE), required=True)
    plan.add_argument("--dir", type=Path, required=True)
    plan.add_argument("--version", required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--expect", type=int, default=4)
    plan.add_argument("--json-base", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "source":
            check_source(args.index, args.run, args.repo, args.sha, args.main_ref)
            print(f"pypiindex: run {args.run} is a valid {args.index} source")
            return 0
        upload = plan_upload(
            args.index, args.dir, args.version, args.out, args.expect, args.json_base
        )
    except ReleaseError as err:
        print(f"pypiindex: {err}", file=sys.stderr)
        return 1
    print(f"pypiindex: {len(upload)} wheel(s) to upload to {args.index}: {upload}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"upload={len(upload)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
