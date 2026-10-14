"""Prove aptserver.py leaves the rest of the zone alone and repeats cleanly.

`tool/tasks/aptserver.py` creates the bucket, the Worker and the route that
serve urna.dev/apt. This suite replaces its one HTTP call with a recorder
standing in for the Cloudflare API:

- happy path: an empty account gets the bucket, the Worker (bound to the
  bucket, workers.dev off) and the route `urna.dev/apt/*`; a second run
  keeps the bucket and the route and creates nothing;
- error path: a route of the Worker other than `urna.dev/apt/*`, or
  `urna.dev/apt/*` already going to another script, is refused, never
  rewritten; an API error names what failed;
- edge case: the upload carries `pkgs/linux/apt/worker.js` byte for byte as
  an ES module, so the deployed Worker is the reviewed one.

Run: python tool/tests/test_aptserver.py
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aptserver", REPO / "tool" / "tasks" / "aptserver.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)

OK = {"success": True, "result": {}}


class Cloud:
    def __init__(self, routes=None, bucket=False):
        self.routes = list(routes or [])
        self.bucket = bucket
        self.calls: list[tuple[str, str]] = []
        self.bodies: dict[str, bytes] = {}

    def __call__(self, token, method, path, body=None, kind="application/json"):
        self.calls.append((method, path))
        if body is not None:
            self.bodies[f"{method} {path}"] = body
        if path.endswith(f"/r2/buckets/{ap.BUCKET}") and method == "GET":
            return (200, OK) if self.bucket else (404, {"success": False, "errors": []})
        if path.endswith("/r2/buckets") and method == "POST":
            self.bucket = True
            return 200, OK
        if path.endswith("/workers/routes") and method == "GET":
            return 200, {"success": True, "result": self.routes}
        if path.endswith("/workers/routes") and method == "POST":
            self.routes.append(json.loads(body))
            return 200, OK
        if "/workers/scripts/" in path:
            return 200, OK
        return 500, {"success": False, "errors": [{"message": "unexpected call"}]}


def test_first_run_creates_and_second_run_keeps() -> None:
    cloud = Cloud()
    ap.call = cloud
    assert ap.bucket("t", "a") == "created"
    assert ap.worker("t", "a") == "uploaded"
    assert ap.route("t", "z") == "created"
    assert cloud.routes == [{"pattern": "urna.dev/apt/*", "script": "urna-apt"}]
    off = cloud.bodies["POST /accounts/a/workers/scripts/urna-apt/subdomain"]
    assert json.loads(off) == {"enabled": False}
    cloud.calls.clear()
    assert ap.bucket("t", "a") == "kept" and ap.route("t", "z") == "kept"
    assert all(method == "GET" for method, _ in cloud.calls), cloud.calls
    print("happy (created once, kept on the second run, workers.dev off): OK")


def test_other_routes_are_refused() -> None:
    for routes, needle in [
        ([{"pattern": "urna.dev/*", "script": "urna-apt"}], "also has the routes"),
        ([{"pattern": "urna.dev/apt/*", "script": "something-else"}], "already goes to"),
    ]:
        ap.call = Cloud(routes=routes)
        try:
            ap.route("t", "z")
        except ap.ServerError as err:
            assert needle in str(err), err
        else:
            raise AssertionError(f"{routes} was accepted")
    ap.call = lambda *a, **k: (403, {"success": False, "errors": [{"message": "no permission"}]})
    try:
        ap.bucket("t", "a")
    except ap.ServerError as err:
        assert "read bucket urna-apt: no permission" in str(err), err
    else:
        raise AssertionError("an API error was accepted")
    print("error (a wider route, a taken route, an API error): OK")


def test_upload_is_the_reviewed_module() -> None:
    cloud = Cloud(bucket=True)
    ap.call = cloud
    ap.worker("t", "a")
    body = cloud.bodies["PUT /accounts/a/workers/scripts/urna-apt"]
    source = (REPO / "pkgs/linux/apt/worker.js").read_bytes()
    assert source in body and b"application/javascript+module" in body
    meta = body.split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0]
    bindings = json.loads(meta)["bindings"]
    assert bindings == [{"type": "r2_bucket", "name": "APT", "bucket_name": "urna-apt"}], bindings
    print("edge (worker.js byte for byte, the bucket bound as APT): OK")


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    real = ap.call
    try:
        for test in tests:
            test()
    finally:
        ap.call = real
    print(f"aptserver: {len(tests)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
