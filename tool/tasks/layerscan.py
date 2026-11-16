"""check the import contracts of pyproject.toml ([tool.layerscan]) on the urna package.

what `lint-imports` (import-linter) checked before #643, with the standard
library: every module under rust/bridge/python/<root package> is parsed with
`ast` and each `import x` / `from x import y` it holds, at any depth (inside
functions too), is matched against the contracts. a contract

    [[tool.layerscan.contracts]]
    name = "urna does not import tool/bench"
    type = "forbidden"
    source_modules = ["urna"]
    forbidden_modules = ["baselines", ...]

fails when a module of a source package imports a forbidden module or one
of its submodules. "forbidden" is the one contract type the project uses; a
contract of another type is an error, not a silent pass. like import-linter,
it sees no module named in a string (`importlib.import_module("x")`).

run:  python tool/tasks/layerscan.py        (exit 1 and one line per import)
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rust" / "bridge" / "python"


def module_name(path: Path, base: Path) -> str:
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports(path: Path, name: str) -> list[tuple[int, str]]:
    """(line, absolute module) for every import in the file; relative ones
    resolve against the module's package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            found.append((node.lineno, target))
            # `from x import y` may import the submodule x.y
            found += [(node.lineno, f"{target}.{alias.name}") for alias in node.names]
    return found


def matches(module: str, prefixes: list[str]) -> str | None:
    for p in prefixes:
        if module == p or module.startswith(p + "."):
            return p
    return None


def check(config: dict, base: Path = SOURCE) -> list[str]:
    problems = []
    for contract in config.get("contracts", []):
        if contract.get("type") != "forbidden":
            problems.append(f"contract {contract.get('name')!r}: type {contract.get('type')!r}")
            continue
        for source in contract["source_modules"]:
            top = base / source.replace(".", "/")
            files = sorted(top.rglob("*.py")) if top.is_dir() else [top.with_suffix(".py")]
            for path in files:
                name = module_name(path, base)
                if not matches(name, [source]):
                    continue
                for line, target in imports(path, name):
                    hit = matches(target, contract["forbidden_modules"])
                    if hit:
                        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                        problems.append(
                            f"{rel}:{line}: {name} imports {target} "
                            f"({contract['name']}: {hit} is forbidden)"
                        )
    return problems


def main() -> int:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["layerscan"]
    problems = check(config)
    for p in problems:
        print(p)
    if problems:
        return 1
    n = len(config.get("contracts", []))
    print(f"layerscan: {n} import contract(s) kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
