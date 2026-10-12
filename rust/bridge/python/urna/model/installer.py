"""installer.py - fetch one catalog model into the local hugging face cache.

the second half of the one install operation `urna setup` and the explorer
share (`rust/clitui/src/tui/setup/models.rs`): the rust side installs the
packages into the managed venv, then runs this script with that venv's
python to fetch the weights and prove them.

  <py> installer.py plan  <name|embedding_model> [--expect-hash H]
  <py> installer.py fetch <name|embedding_model> [--expect-hash H]
                                                  [--allow-remote-code]

plan prints one JSON line: the entry, the bytes still to fetch, whether the
pinned revision is already cached and which packages do not import here.
it never touches the network.

fetch downloads exactly the catalog's files at its pinned revision into the
shared cache (`$HF_HOME/hub`, the cache the query embedders read), then
fingerprints them and refuses unless the result is the catalog's
model_hash (and the corpus's, with --expect-hash). an open_clip model's
fingerprint is over the tensors it loads, so its check loads it. a
sentence-transformers backend model (`st_multimodal`) stamps its corpora
with a hash that also names the dtype it ran in, which depends on the
device; its --expect-hash is checked by the query setup runs next. the network is used only
with URNA_ALLOW_DOWNLOAD=1, which the caller sets after the user confirmed
the download; a model that runs repo code also needs --allow-remote-code, a
separate consent, checked before anything is fetched. progress goes to
stdout as `urna-progress: <done> <total> <file>` lines.

exit codes: 0 ok, 2 usage, 3 download not allowed (`urna-fetch: <repo>`),
4 a package missing (`urna-needs: <spec>`), 5 remote code not allowed
(`urna-consent: remote-code <name>`), 6 a hash or revision that does not
match, 7 a model the catalog does not offer, 8 the download failed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# run as a file (rust/bridge/python/ or the payload python/), the folder that
# holds urna/ goes first; imported as a module, the package is already there.
if not __package__:
    sys.path.insert(0, str(HERE.parents[1]))

CATALOG = HERE / "catalogue.json"


class Refused(Exception):
    def __init__(self, code: int, msg: str, tag: str | None = None):
        super().__init__(msg)
        self.code = code
        self.tag = tag


def load_catalog(path: Path = CATALOG) -> dict:
    return json.loads(path.read_text())


def find(catalog: dict, name: str) -> dict:
    """The offered entry for a catalog name or a manifest model name."""
    for m in catalog["models"]:
        if name in (m["name"], m["embedding_model"]):
            return m
    for e in catalog["excluded"]:
        if name in (e["name"], e["embedding_model"]):
            raise Refused(7, f"{e['name']} is not offered for install: {e['reason']}")
    offered = ", ".join(m["name"] for m in catalog["models"]) or "none"
    raise Refused(7, f"{name} is not in the model catalog (offered: {offered})")


def hub_dir() -> Path:
    """The cache the query embedders read (modelhash.hf_cache_snapshot):
    $HF_HOME/hub, else ~/.cache/huggingface/hub."""
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def repo_dir(entry: dict) -> Path:
    return hub_dir() / f"models--{entry['repo'].replace('/', '--')}"


def snapshot(entry: dict) -> Path:
    return repo_dir(entry) / "snapshots" / entry["revision"]


def missing_files(entry: dict) -> list[dict]:
    snap = snapshot(entry)
    return [f for f in entry["files"] if not (snap / f["path"]).is_file()]


def missing_packages(entry: dict) -> list[str]:
    if all(importlib.util.find_spec(m) is not None for m in entry["imports"]):
        return []
    return list(entry["packages"])


def check_expected(entry: dict, expect: str | None) -> None:
    if entry["kind"] == "st_multimodal":
        return
    if expect and expect != entry["model_hash"]:
        raise Refused(
            6,
            f"{entry['name']} at revision {entry['revision'][:12]} fingerprints to "
            f"{entry['model_hash']}, but the corpus was built with {expect}: installing "
            "it would not make this corpus askable",
        )


def plan(entry: dict, expect: str | None) -> dict:
    check_expected(entry, expect)
    todo = missing_files(entry)
    return {
        "name": entry["name"],
        "repo": entry["repo"],
        "revision": entry["revision"],
        "bytes": entry["bytes"],
        "missing_bytes": sum(f["size"] for f in todo),
        "cached": not todo,
        "packages_missing": missing_packages(entry),
        "remote_code": entry["remote_code"],
        "model_hash": entry["model_hash"],
        "python": sys.executable,
        "cache": str(hub_dir()),
    }


def _progress(done: int, total: int, path: str) -> None:
    print(f"urna-progress: {done} {total} {path}", flush=True)


def _reporter(base: int, size: int, total: int, path: str):
    """A progress class for hf_hub_download: the hub client feeds it the
    bytes of one file over plain http and over xet alike (xet holds a file
    in memory until it is whole, so a partial file on disk says nothing).
    it draws no bar; it prints an urna-progress line every MB.

    xet opens two bars for one file with this class, the reconstructed
    bytes and the transferred (compressed, fewer) bytes, each counting from
    zero. the instances share what was shown, so the lines never go back,
    and a count is capped at the file's size."""
    from huggingface_hub.utils import tqdm as hub_tqdm

    shown = {"at": -1}

    class Report(hub_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = io.StringIO()
            super().__init__(*args, **kwargs)
            self.n = 0

        def update(self, n=1):
            self.n += n or 0
            at = base + min(int(self.n), size)
            if at - shown["at"] >= 1 << 20:
                shown["at"] = at
                _progress(at, total, path)

        def refresh(self, *args, **kwargs):
            pass

        def close(self):
            pass

    return Report


def download(entry: dict) -> None:
    # the progress lines replace the client's bars; its warnings stay out.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    if importlib.util.find_spec("huggingface_hub") is None:
        raise Refused(4, "fetching a model needs huggingface_hub", "urna-needs: huggingface_hub")
    from huggingface_hub import hf_hub_download

    total = entry["bytes"]
    todo = {f["path"] for f in missing_files(entry)}
    done = total - sum(f["size"] for f in entry["files"] if f["path"] in todo)
    for f in entry["files"]:
        if f["path"] not in todo:
            continue
        _progress(done, total, f["path"])
        try:
            hf_hub_download(
                entry["repo"],
                f["path"],
                revision=entry["revision"],
                cache_dir=str(hub_dir()),
                tqdm_class=_reporter(done, f["size"], total, f["path"]),
            )
        except Exception as e:
            raise Refused(8, f"downloading {f['path']} from {entry['repo']} failed: {e}") from e
        done += f["size"]
        _progress(done, total, f["path"])


def pin_main_ref(entry: dict) -> None:
    """Point refs/main at the pinned revision when the cache has no main yet:
    sentence-transformers and the fingerprint load what refs/main names. a
    main that already names another revision belongs to whoever fetched it;
    it is left alone and the run fails naming both."""
    ref = repo_dir(entry) / "refs" / "main"
    if not ref.is_file():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(entry["revision"])
        return
    held = ref.read_text().strip()
    if held != entry["revision"]:
        raise Refused(
            6,
            f"the cache's main for {entry['repo']} is {held[:12]}, the catalog pins "
            f"{entry['revision'][:12]}; queries load main, so they would not get the "
            f"pinned model. the pinned files are at {snapshot(entry)} (--model-path)",
        )


def verify(entry: dict) -> str:
    snap = snapshot(entry)
    for pin in entry["remote_code_hashes"]:
        got = hashlib.sha256((snap / pin["path"]).read_bytes()).hexdigest()
        if got != pin["sha256"]:
            raise Refused(6, f"{pin['path']} sha256 {got} is not the reviewed {pin['sha256']}")
    if entry["kind"] == "open_clip":
        got = open_clip_hash(entry, snap)
    else:
        from urna.model.modelhash import compute_model_fingerprint, fingerprint_to_model_hash

        fp = compute_model_fingerprint(snap, model_id=entry["embedding_model"])
        got = fingerprint_to_model_hash(fp)
    if got != entry["model_hash"]:
        raise Refused(
            6,
            f"{entry['name']} at {entry['revision'][:12]} fingerprints to {got}, "
            f"not the catalog's {entry['model_hash']}",
        )
    return got


def open_clip_hash(entry: dict, snap: Path) -> str:
    """The model_hash an open_clip preset loads to from `snap`: the hash of
    its tensors and preprocess, the one its corpora carry."""
    from urna.embed.visionemb import ImageEmbedder
    from urna.model.presetmap import get_preset

    preset = get_preset(entry["name"])
    emb = ImageEmbedder(
        model_id=preset.model_id,
        pretrained=preset.pretrained,
        device="cpu",
        snapshot=snap,
        weights=preset.weights_file or None,
    )
    return emb.model_hash


def fetch(entry: dict, expect: str | None, allow_remote_code: bool) -> dict:
    check_expected(entry, expect)
    if entry["remote_code"] and not allow_remote_code:
        raise Refused(
            5,
            f"{entry['name']} runs code from its model repo; that needs its own consent",
            f"urna-consent: remote-code {entry['name']}",
        )
    if missing_files(entry):
        if os.environ.get("URNA_ALLOW_DOWNLOAD") != "1":
            raise Refused(
                3,
                f"{entry['repo']} is not in the local cache and downloads were not confirmed",
                f"urna-fetch: {entry['repo']}",
            )
        download(entry)
    pin_main_ref(entry)
    model_hash = verify(entry)
    return {"name": entry["name"], "model_hash": model_hash, "snapshot": str(snapshot(entry))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("plan", "fetch"))
    ap.add_argument("model")
    ap.add_argument("--expect-hash")
    ap.add_argument("--allow-remote-code", action="store_true")
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    args = ap.parse_args(argv)
    try:
        entry = find(load_catalog(args.catalog), args.model)
        if args.action == "plan":
            out = plan(entry, args.expect_hash)
        else:
            out = fetch(entry, args.expect_hash, args.allow_remote_code)
    except Refused as e:
        print(f"error: {e}", file=sys.stderr)
        if e.tag:
            print(e.tag, file=sys.stderr)
        return e.code
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
