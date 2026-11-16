"""Prove the import check holds urna to the contracts of pyproject.toml.

`tool/tasks/layerscan.py` runs in fullcheck.sh and gatecheck.yml, in place
of import-linter. this suite runs it on the real checkout, then on a small
throwaway package, one drift per case:

- happy path: the checkout keeps its contract; a package that imports only
  the standard library and itself passes;
- error path: `import benchgate`, `from presetrun import x`, an import inside
  a function, a forbidden submodule (`baselines.load`) each fail with the
  file and the line; a contract of a type the script does not know fails;
- edge case: a name that only starts like a forbidden one (`benchgates`)
  passes; a relative import resolves inside the package and passes.

Run: python tool/tests/test_layerscan.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tool" / "tasks" / "layerscan.py"
spec = importlib.util.spec_from_file_location("layerscan", SCRIPT)
layerscan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layerscan)

CONTRACT = {
    "contracts": [
        {
            "name": "pkg does not import the bench",
            "type": "forbidden",
            "source_modules": ["pkg"],
            "forbidden_modules": ["benchgate", "presetrun", "baselines"],
        }
    ]
}


def tree(base: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return base


def test_the_checkout_keeps_its_contract() -> None:
    assert layerscan.main() == 0
    print("happy (the checkout keeps the pyproject contract): OK")


def test_a_clean_package_passes() -> None:
    with tempfile.TemporaryDirectory() as d:
        base = tree(
            Path(d),
            {
                "pkg/__init__.py": "import os\nfrom . import sub\n",
                "pkg/sub/__init__.py": "from ..sub import mod\nimport benchgates\n",
                "pkg/sub/mod.py": "from .. import sub\n",
            },
        )
        assert layerscan.check(CONTRACT, base) == []
    print("happy + edge (stdlib, relative imports, a look-alike name): OK")


def test_each_forbidden_import_is_named() -> None:
    with tempfile.TemporaryDirectory() as d:
        base = tree(
            Path(d),
            {
                "pkg/__init__.py": "",
                "pkg/a.py": "import benchgate\n",
                "pkg/b.py": "from presetrun import run\n",
                "pkg/c.py": "def f():\n    import baselines.load\n",
            },
        )
        problems = layerscan.check(CONTRACT, base)
    joined = "\n".join(problems)
    for needle in (
        "a.py:1: pkg.a imports benchgate",
        "b.py:1: pkg.b imports presetrun",
        "c.py:2: pkg.c imports baselines.load",
    ):
        assert needle in joined, joined
    print("error (direct, from-import, nested and submodule imports are named): OK")


def test_an_unknown_contract_type_is_an_error() -> None:
    contract = {"contracts": [{"name": "layers", "type": "layers"}]}
    assert layerscan.check(contract, REPO) == ["contract 'layers': type 'layers'"]
    print("error (a contract type the script does not check fails): OK")


def main() -> None:
    test_the_checkout_keeps_its_contract()
    test_a_clean_package_passes()
    test_each_forbidden_import_is_named()
    test_an_unknown_contract_type_is_an_error()
    print("layerscan: 4 cases passed")


if __name__ == "__main__":
    sys.exit(main())
