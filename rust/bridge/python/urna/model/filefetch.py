"""filefetch.py - one pinned file from the hugging face hub, by stdlib urllib.

what the model installer, the preset registry and the benchmark corpus
fetch used huggingface_hub for (#643): GET
`https://huggingface.co/<repo>/resolve/<revision>/<path>` (datasets under
`/datasets/`), follow the hub's redirect to its storage over https only,
stream the body into the cache layout the hub client writes and the query
embedders read (`<hub>/<models|datasets>--org--name/snapshots/<rev>/<path>`,
as a plain file, which the hub libraries accept), and check it before it
takes the final name:

- the size, when the caller knows it (the model catalog does);
- the sha256 the caller pins, else the one the hub states for a large
  (LFS/xet) file in `X-Linked-Etag`, else the git blob sha1 it states in
  `ETag` for a small file. a file the hub gives no hash for is refused.

nothing here reads HF_HUB_OFFLINE: every caller fetches only after its own
explicit opt-in (URNA_ALLOW_DOWNLOAD=1), as before. no token is sent, so
gated repos are out of reach (urllib would forward one to the storage host
the hub redirects to).
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

ENDPOINT = "https://huggingface.co"
TIMEOUT = 60  # seconds per connect or read; a stalled transfer fails, it does not hang
CHUNK = 1 << 20
_HEX = re.compile(r"^[0-9a-f]+$")


class FetchError(Exception):
    """The file could not be fetched or did not check out."""


def hub_dir() -> Path:
    """$HF_HOME/hub, else ~/.cache/huggingface/hub: the cache the embedders read."""
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def snapshot_dir(
    repo: str, revision: str, repo_type: str = "model", hub: Path | None = None
) -> Path:
    kind = {"model": "models", "dataset": "datasets"}[repo_type]
    return (hub or hub_dir()) / f"{kind}--{repo.replace('/', '--')}" / "snapshots" / revision


def resolve_url(repo: str, revision: str, path: str, repo_type: str = "model") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    quote = urllib.parse.quote
    return f"{ENDPOINT}/{prefix}{repo}/resolve/{quote(revision, safe='')}/{quote(path)}"


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Follows redirects to https only, and keeps the hub's own headers (the
    first response), which carry the file's hash; the storage host's do not."""

    def __init__(self) -> None:
        super().__init__()
        self.hub_headers = None

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise FetchError(f"the hub redirected to a non-https url: {newurl}")
        if self.hub_headers is None:
            self.hub_headers = headers
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _unquote(etag: str | None) -> str:
    return (etag or "").strip().removeprefix("W/").strip('"').lower()


def _expected(headers, pinned: str | None) -> tuple[str, str]:
    """(algorithm, hex digest) the body must hash to."""
    if pinned:
        return "sha256", pinned.lower().removeprefix("sha256:")
    linked = _unquote(headers.get("X-Linked-Etag"))
    if len(linked) == 64 and _HEX.match(linked):
        return "sha256", linked
    etag = _unquote(headers.get("ETag"))
    if len(etag) == 40 and _HEX.match(etag):
        return "gitsha1", etag
    raise FetchError("the hub stated no hash for the file and none is pinned")


def fetch_file(
    repo: str,
    path: str,
    revision: str,
    *,
    repo_type: str = "model",
    size: int | None = None,
    sha256: str | None = None,
    hub: Path | None = None,
    progress: Callable[[int], None] | None = None,
    timeout: float = TIMEOUT,
) -> Path:
    """Fetch `path` of `repo` at `revision` into the hub cache and return it.

    `progress` gets the bytes received so far, about once per MB."""
    dest = snapshot_dir(repo, revision, repo_type, hub) / path
    url = resolve_url(repo, revision, path, repo_type)
    redirects = _HttpsOnly()
    opener = urllib.request.build_opener(redirects)
    request = urllib.request.Request(url, headers={"User-Agent": "urna-filefetch"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with opener.open(request, timeout=timeout) as resp:
            algo, want = _expected(redirects.hub_headers or resp.headers, sha256)
            declared = (redirects.hub_headers or resp.headers).get("X-Linked-Size")
            length = int(declared) if declared and declared.isdigit() else None
            length = length if length is not None else resp.headers.get("Content-Length")
            digest = hashlib.sha256() if algo == "sha256" else hashlib.sha1()
            got = 0
            body = bytearray() if algo == "gitsha1" else None
            with part.open("wb") as out:
                while block := resp.read(CHUNK):
                    out.write(block)
                    got += len(block)
                    if body is not None:
                        body += block
                    else:
                        digest.update(block)
                    if progress:
                        progress(got)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
        part.unlink(missing_ok=True)
        raise FetchError(f"{url}: {e}") from e
    try:
        expected_size = size if size is not None else (int(length) if length else None)
        if expected_size is not None and got != expected_size:
            raise FetchError(f"{url}: got {got} bytes, expected {expected_size}")
        if body is not None:
            digest.update(b"blob %d\0" % got)
            digest.update(body)
        if digest.hexdigest() != want:
            raise FetchError(f"{url}: {algo} {digest.hexdigest()} is not the expected {want}")
    except FetchError:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return dest
