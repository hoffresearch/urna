"""Prove each channel's manifest comes from the release, escaped, or not at all.

`tool/tasks/pkgrender.py` fills the templates in `pkgs/` from a release's
`dist-manifest.json` and `sha256.sum`. This suite renders the checked-in
templates against v0.5.4's values (the archives, their layout and their
SHA-256 as that release published them) in a scratch root that holds a copy
of `pkgs/` and a Cargo.toml with `[workspace.metadata.pkgs]`:

- happy path: every channel renders; each URL is the release's and each
  hash is `sha256.sum`'s; the scoop manifest and the nuspec parse and keep
  scoop's own `$version`; nfpm gets one config per architecture with the
  archive's folder; no placeholder is left in any file;
- error path: an archive missing from `sha256.sum`, a tag that is not
  vX.Y.Z, a release of another version, a summary over 80 characters, a
  template naming an unknown value (in the second file of a channel) and an
  empty template each stop the render with the reason, and no file of the
  channel is written;
- edge case: quotes, `&`, `<` and a newline in the texts stay inside their
  JSON, YAML and XML strings; an archive URL of another repository or of
  another tag is refused;
  the PowerShell script gets no text at all.

Run: python tool/tests/test_pkgrender.py
"""

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pkgrender", REPO / "tool/tasks/pkgrender.py")
pkgrender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkgrender)

BASE = "https://github.com/hoffresearch/urna/releases/download/v0.5.4"
# what v0.5.4 published: archive -> (target, executable path, SHA-256).
ARCHIVES = {
    "urna-x86_64-unknown-linux-musl.tar.xz": (
        "x86_64-unknown-linux-musl",
        "urna",
        "6df332f46f2ddb9c4de27bb2772869418d453622ea9ea7f319a3d269ef6055db",
    ),
    "urna-aarch64-unknown-linux-musl.tar.xz": (
        "aarch64-unknown-linux-musl",
        "urna",
        "4263840f54b2865dcd74051f389d2d5e7dc55799be4a5601de5e8dda7872ce3b",
    ),
    "urna-x86_64-apple-darwin.tar.xz": (
        "x86_64-apple-darwin",
        "urna",
        "7d937b32f36a2c9ed5fbcbaedac099514a2be777dc1161bb0458a6cfe76037e7",
    ),
    "urna-aarch64-apple-darwin.tar.xz": (
        "aarch64-apple-darwin",
        "urna",
        "c6bbbf991c4a37b0e91082bdae6a513cbc73bfb0b4949f3e4b18951285cf1662",
    ),
    "urna-x86_64-pc-windows-msvc.zip": (
        "x86_64-pc-windows-msvc",
        "urna.exe",
        "f0622e3de926a2c9176f07c8fe23afddb2cb48ad83ce140d69eefee0321cd94a",
    ),
}
META = """
[workspace.package]
version = "0.5.4"
license = "MIT"
repository = "https://github.com/hoffresearch/urna"
homepage = "https://urna.dev"

[workspace.metadata.pkgs]
summary = {summary}
description = {description}
tags = ["vector-database", "offline", "citations"]
"""
PLAIN = ("Offline vector database in one file", "Build a corpus once, ask it anywhere.")
TRICKY = ("Files & <citations>", 'One "quoted" line,\nthen a <tag> & a backslash \\ here.')


def manifest(tag="v0.5.4", version="0.5.4", base=BASE):
    artifacts = {
        name: {
            "kind": "executable-zip",
            "target_triples": [triple],
            "assets": [{"name": "urna", "path": exe, "kind": "executable"}],
        }
        for name, (triple, exe, _) in ARCHIVES.items()
    }
    owner, _, rest = base.removeprefix("https://github.com/").partition("/")
    repo, _, path = rest.partition("/")
    hosting = {
        "artifact_base_url": "https://github.com",
        "artifact_download_path": f"/{owner}/{repo}/{path}",
    }
    return {
        "announcement_tag": tag,
        "releases": [{"app_name": "urna", "app_version": version, "hosting": {"github": hosting}}],
        "artifacts": artifacts,
    }


def scratch(tmp: Path, texts=PLAIN, drop=None, data=None) -> tuple[Path, Path, Path]:
    root = tmp / "root"
    shutil.copytree(REPO / "pkgs", root / "pkgs", ignore=shutil.ignore_patterns("stage", "wheel"))
    summary, description = texts
    (root / "Cargo.toml").write_text(
        META.format(summary=json.dumps(summary), description=json.dumps(description))
    )
    man = tmp / "dist-manifest.json"
    man.write_text(json.dumps(data or manifest()))
    sums = tmp / "sha256.sum"
    sums.write_text("".join(f"{h} *{n}\n" for n, (_, _, h) in ARCHIVES.items() if n != drop))
    return root, man, sums


def run(root: Path, man: Path, sums: Path, out: Path, channel: str, *extra: str) -> int:
    args = [
        "render",
        "--channel",
        channel,
        "--manifest",
        str(man),
        "--sums",
        str(sums),
        "--out",
        str(out),
        *extra,
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        return pkgrender.main([*args, "--root", str(root)])


def rendered(out: Path) -> dict[str, str]:
    return {str(p.relative_to(out)): p.read_text() for p in sorted(out.rglob("*")) if p.is_file()}


def case_every_channel(tmp: Path):
    root, man, sums = scratch(tmp)
    out = tmp / "out"
    for channel in pkgrender.CHANNELS:
        assert run(root, man, sums, out, channel) == 0, channel
    files = rendered(out)
    expected = {
        "scoop/urna.json",
        "wingt/HoffResearch.Urna.yaml",
        "wingt/HoffResearch.Urna.installer.yaml",
        "wingt/HoffResearch.Urna.locale.en-US.yaml",
        "conda/recipe.yaml",
        "choco/urna.nuspec",
        "choco/tools/chocolateyInstall.ps1",
        "linux/dnf/urna.spec",
        "linux/nfpm-amd64.yaml",
        "linux/nfpm-arm64.yaml",
    }
    assert set(files) == expected, sorted(files)
    for path, text in files.items():
        assert "${" not in text, f"{path} kept a placeholder"
    win = "urna-x86_64-pc-windows-msvc.zip"
    scoop = json.loads(files["scoop/urna.json"])
    assert scoop["version"] == "0.5.4"
    assert scoop["architecture"]["64bit"] == {"url": f"{BASE}/{win}", "hash": ARCHIVES[win][2]}
    assert "/v$version/" in scoop["autoupdate"]["architecture"]["64bit"]["url"], (
        "scoop's $version lost"
    )
    nuspec = ET.fromstring(files["choco/urna.nuspec"])
    ns = {"n": "http://schemas.microsoft.com/packaging/2015/06/nuspec.xsd"}
    assert nuspec.find("n:metadata/n:version", ns).text == "0.5.4"
    assert f"'{BASE}/{win}'" in files["choco/tools/chocolateyInstall.ps1"]
    assert f"InstallerSha256: {ARCHIVES[win][2]}" in files["wingt/HoffResearch.Urna.installer.yaml"]
    for name, (_, _, digest) in ARCHIVES.items():
        assert f"url: {BASE}/{name}\n      sha256: {digest}\n" in files["conda/recipe.yaml"], name
    musl = [n for n in ARCHIVES if "linux-musl" in n]
    for name in musl:
        assert f"{BASE}/{name}" in files["linux/dnf/urna.spec"], name
    for arch, name in (("amd64", musl[0]), ("arm64", musl[1])):
        cfg = files[f"linux/nfpm-{arch}.yaml"]
        assert f"arch: {arch}\n" in cfg and f"src: {name.split('.tar.')[0]}/urna\n" in cfg, arch
    print(f"case 1 ({len(files)} files, every URL and hash the release's): OK")


def case_refused(tmp: Path):
    big = ("x" * 81, PLAIN[1])
    cases = [
        (
            "no hash",
            {"drop": "urna-x86_64-pc-windows-msvc.zip"},
            "no SHA-256 for urna-x86_64-pc-windows-msvc.zip",
        ),
        ("bad tag", {"data": manifest(tag="latest")}, "is not vX.Y.Z"),
        ("other version", {"data": manifest(version="0.5.3")}, "no urna release at 0.5.4"),
        ("long summary", {"texts": big}, "over 80 characters"),
    ]
    for label, kw, why in cases:
        sub = tmp / label.replace(" ", "-")
        root, man, sums = scratch(sub, **kw)
        assert_refused(root, man, sums, sub / "out", "scoop", why)
    sub = tmp / "unknown"
    root, man, sums = scratch(sub)
    # the second of choco's two files breaks: the first is not written either.
    (root / "pkgs/choco/tools/chocolateyInstall.ps1").write_text("'${versoin}'\n")
    assert_refused(root, man, sums, sub / "out", "choco", "unknown or malformed placeholder")
    sub = tmp / "other-tag"
    root2, man2, sums2 = scratch(sub)
    assert_refused(
        root2,
        man2,
        sums2,
        sub / "out",
        "choco",
        "is the release of v0.5.4, not v0.5.5",
        "--tag",
        "v0.5.5",
    )
    assert run(root2, man2, sums2, sub / "ok", "choco", "--tag", "v0.5.4") == 0
    (root / "pkgs/scoop/urna.json").write_text("")
    assert_refused(root, man, sums, sub / "out", "scoop", "pkgs/scoop/urna.json is empty")
    print(
        "case 2 (no hash, bad tag, other version, another job's tag, long text, unknown name, empty file): OK"
    )


def assert_refused(root, man, sums, out, channel, why, *extra):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = run(root, man, sums, out, channel, *extra)
    assert code == 1 and why in err.getvalue(), (why, err.getvalue())
    assert not out.exists(), f"{out} was written"


def case_escaping(tmp: Path):
    root, man, sums = scratch(tmp, texts=TRICKY)
    out = tmp / "out"
    for channel in ("scoop", "choco", "wingt", "conda"):
        assert run(root, man, sums, out, channel) == 0, channel
    files = rendered(out)
    summary, description = TRICKY
    flat = " ".join(description.split())
    assert json.loads(files["scoop/urna.json"])["description"] == summary
    meta = ET.fromstring(files["choco/urna.nuspec"])[0]
    got = {el.tag.split("}")[1]: el.text for el in meta}
    assert got["summary"] == summary and got["description"] == flat, got
    locale = files["wingt/HoffResearch.Urna.locale.en-US.yaml"]
    assert f"Description: {json.dumps(flat)}\n" in locale, "winget description"
    assert f"summary: {json.dumps(summary)}\n" in files["conda/recipe.yaml"]
    ps1 = files["choco/tools/chocolateyInstall.ps1"]
    assert "citations" not in ps1 and "quoted" not in ps1, "text in the PowerShell script"

    sub = tmp / "elsewhere"
    root, man, sums = scratch(
        sub, data=manifest(base="https://github.com/someone/else/releases/download/v0.5.3")
    )
    assert_refused(root, man, sums, sub / "out", "scoop", "is not under this release")
    sub = tmp / "othertag"
    root, man, sums = scratch(sub, data=manifest(base=BASE.replace("v0.5.4", "v0.5.3")))
    assert_refused(root, man, sums, sub / "out", "scoop", "is not under this release")
    print(
        "case 3 (texts escaped for JSON, YAML, XML, none in PowerShell, foreign URLs refused): OK"
    )


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        case_every_channel(tmp / "1")
        case_refused(tmp / "2")
        case_escaping(tmp / "3")
    print("pkgrender: 3 cases passed")


if __name__ == "__main__":
    main()
