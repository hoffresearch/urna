"""the corpus fullcheck measures: one pinned file of the benchmark dataset.

`fakenews.urna` (MiniLM 384d, exact preset) of the fakenews-ptbr-urna-benchmark
dataset on hugging face, at a fixed commit of the dataset, checked against
the sha-256 its SHA256SUMS lists. it lives in the hugging face cache
($HF_HOME/hub, else ~/.cache/huggingface/hub), outside the repo, where the
model installer keeps the models.

the download happens only with URNA_ALLOW_DOWNLOAD=1, the key the model
fetch uses; without it and without the file, the run stops and names the
command that fetches it. a file in the cache whose sha-256 is not the
pinned one is refused, never measured.

    fetch                    print the path of the checked corpus, fetching it if allowed
    path                     print where the corpus is (or would be) in the cache
    match BASELINE CORPUS    check that BASELINE was measured on CORPUS: its
                             baseline_file_hash is the sha-256 of CORPUS

exit codes: 2 usage, 3 corpus not cached and downloads not confirmed,
4 huggingface_hub missing, 6 cached corpus not the pinned one, 8 download
failed, 9 baseline missing or unreadable, 10 baseline measured on another
corpus.

run:  python tool/tasks/benchdata.py fetch
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = "brennercruvinel/fakenews-ptbr-urna-benchmark"
REVISION = "b58d0c8be8dbc3b8b387a8b9047a1abbf4819f73"
FILE = "release/v0.1/minilm-exact/fakenews.urna"
SHA256 = "4bcb8e0fc38a1c13f9459ed8b1553cc7c5261c85bbb515235c4fe87e34d04cff"


class Refused(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def hub_dir() -> Path:
    """$HF_HOME/hub, else ~/.cache/huggingface/hub: the cache the model
    installer (urna.model.installer.hub_dir) fetches into. repeated here so
    the gate can find the corpus before the extension is built."""
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def path() -> Path:
    """where hf_hub_download puts FILE at REVISION in the cache."""
    repo_dir = "datasets--" + REPO.replace("/", "--")
    return hub_dir() / repo_dir / "snapshots" / REVISION / FILE


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def command() -> str:
    return "URNA_ALLOW_DOWNLOAD=1 python tool/tasks/benchdata.py fetch"


def download() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise Refused(4, "fetching the corpus needs huggingface_hub") from e
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    try:
        hf_hub_download(
            REPO, FILE, repo_type="dataset", revision=REVISION, cache_dir=str(hub_dir())
        )
    except Exception as e:
        raise Refused(8, f"downloading {FILE} from {REPO} failed: {e}") from e


def fetch() -> Path:
    p = path()
    if not p.is_file():
        if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
            raise Refused(
                3,
                f"the benchmark corpus is not in the cache ({p}) and downloads were not "
                f"confirmed. fetch it ({REPO}@{REVISION[:12]}, 85 MB) with:\n"
                f"  {command()}\n"
                "or point URNA_CORPUS at a corpus you have",
            )
        download()
    got = sha256(p)
    if got != SHA256:
        raise Refused(
            6, f"{p} has sha256 {got}, not the pinned {SHA256}; delete it and fetch again"
        )
    return p


def match(baseline: Path, corpus: Path) -> None:
    """refuse a baseline whose numbers come from another corpus: benchgate
    would compare this run against them and pass or fail for no reason."""
    try:
        want = json.loads(baseline.read_text(encoding="utf-8"))["baseline_file_hash"]
    except (OSError, ValueError, KeyError) as e:
        raise Refused(
            9,
            f"no usable baseline at {baseline} ({type(e).__name__}: {e}). URNA_BASELINE "
            "must name the presetrun metrics measured on the corpus this run measures "
            "(tool/bench/reference.json for the pinned one)",
        ) from e
    got = "sha256:" + sha256(corpus)
    if got != want:
        raise Refused(
            10,
            f"{baseline} was measured on the corpus with file hash {want}, and this run "
            f"measures {corpus} ({got}); another corpus needs its own baseline "
            "(run presetrun.py --json on it and point URNA_BASELINE at the result)",
        )


def main(argv: list[str]) -> int:
    if argv[:1] == ["match"] and len(argv) == 3:
        try:
            match(Path(argv[1]), Path(argv[2]))
        except Refused as e:
            print(f"benchdata: {e}", file=sys.stderr)
            return e.code
        return 0
    if argv not in (["fetch"], ["path"]):
        print(__doc__, file=sys.stderr)
        return 2
    if argv == ["path"]:
        print(path())
        return 0
    try:
        print(fetch())
    except Refused as e:
        print(f"benchdata: {e}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
