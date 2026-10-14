"""Create or update what serves urna.dev/apt on Cloudflare, idempotently.

    CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_ZONE_ID=... \
        python tool/tasks/aptserver.py

Three pieces, each left alone when it is already right:

- the R2 bucket `urna-apt`, which aptsigner.yml writes the signed index to;
- the Worker `urna-apt`, from `pkgs/linux/apt/worker.js`, with the bucket
  bound as `APT` and no workers.dev address;
- the route `urna.dev/apt/*` to that Worker, and no other route of it, so
  the rest of the zone keeps reaching the site.

The token needs Workers Scripts, Workers R2 Storage and the zone's Workers
Routes, all as edit. Nothing here reads a file outside the repository, and
nothing printed is a credential.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.cloudflare.com/client/v4"
BUCKET = "urna-apt"
SCRIPT = "urna-apt"
PATTERN = "urna.dev/apt/*"
COMPATIBILITY = "2026-10-01"


class ServerError(Exception):
    pass


def call(
    token: str, method: str, path: str, body: bytes | None = None, kind: str = "application/json"
):
    request = urllib.request.Request(f"{API}{path}", data=body, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", kind)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        try:
            data = json.load(err)
        except ValueError:
            data = {}
        return err.code, data


def need(status: int, data: dict, what: str) -> dict:
    if status >= 300 or not data.get("success", False):
        errors = "; ".join(e.get("message", "") for e in data.get("errors", [])) or f"HTTP {status}"
        raise ServerError(f"{what}: {errors}")
    return data


def bucket(token: str, account: str) -> str:
    status, data = call(token, "GET", f"/accounts/{account}/r2/buckets/{BUCKET}")
    if status == 200:
        return "kept"
    if status != 404:
        need(status, data, f"read bucket {BUCKET}")
    body = json.dumps({"name": BUCKET}).encode()
    need(*call(token, "POST", f"/accounts/{account}/r2/buckets", body), f"create bucket {BUCKET}")
    return "created"


def worker(token: str, account: str) -> str:
    source = (ROOT / "pkgs/linux/apt/worker.js").read_bytes()
    metadata = {
        "main_module": "worker.js",
        "compatibility_date": COMPATIBILITY,
        "bindings": [{"type": "r2_bucket", "name": "APT", "bucket_name": BUCKET}],
    }
    boundary = uuid.uuid4().hex
    parts = [
        (b'name="metadata"', b"application/json", json.dumps(metadata).encode()),
        (b'name="worker.js"; filename="worker.js"', b"application/javascript+module", source),
    ]
    body = (
        b"".join(
            b"--"
            + boundary.encode()
            + b"\r\nContent-Disposition: form-data; "
            + disposition
            + b"\r\nContent-Type: "
            + kind
            + b"\r\n\r\n"
            + content
            + b"\r\n"
            for disposition, kind, content in parts
        )
        + b"--"
        + boundary.encode()
        + b"--\r\n"
    )
    kind = f"multipart/form-data; boundary={boundary}"
    path = f"/accounts/{account}/workers/scripts/{SCRIPT}"
    need(*call(token, "PUT", path, body, kind), f"upload worker {SCRIPT}")
    off = json.dumps({"enabled": False}).encode()
    need(*call(token, "POST", f"{path}/subdomain", off), f"turn off workers.dev for {SCRIPT}")
    return "uploaded"


def route(token: str, zone: str) -> str:
    data = need(*call(token, "GET", f"/zones/{zone}/workers/routes"), "list routes")
    mine = [r for r in data["result"] if r.get("script") == SCRIPT]
    stray = [r["pattern"] for r in mine if r["pattern"] != PATTERN]
    if stray:
        raise ServerError(f"{SCRIPT} also has the routes {stray}: remove them by hand")
    taken = [r for r in data["result"] if r["pattern"] == PATTERN and r.get("script") != SCRIPT]
    if taken:
        raise ServerError(f"{PATTERN} already goes to {taken[0].get('script')!r}")
    if mine:
        return "kept"
    body = json.dumps({"pattern": PATTERN, "script": SCRIPT}).encode()
    need(*call(token, "POST", f"/zones/{zone}/workers/routes", body), f"create route {PATTERN}")
    return "created"


def main() -> int:
    try:
        token = os.environ["CLOUDFLARE_API_TOKEN"]
        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        zone = os.environ["CLOUDFLARE_ZONE_ID"]
    except KeyError as err:
        print(f"aptserver: {err.args[0]} is not set", file=sys.stderr)
        return 2
    try:
        print(f"aptserver: bucket {BUCKET} {bucket(token, account)}")
        print(f"aptserver: worker {SCRIPT} {worker(token, account)}")
        print(f"aptserver: route {PATTERN} {route(token, zone)}")
    except ServerError as err:
        print(f"aptserver: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
