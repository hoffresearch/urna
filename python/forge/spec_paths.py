"""Path expansion over the raw spec document: `${VAR}` then `~/`.

Specs stay machine-portable: a data root is written once as `${MTG_DATA}`
and the operator exports it. Only the braced form expands, so a bare `$`
inside a sql `query` or a text `template` survives, and `{col}` format
placeholders are never touched. An unset or empty variable is a SpecError
naming the dotted key: corpus_sources._Blank would otherwise turn a
leftover `${VAR}` into "$" silently, and an empty root would make
`${MTG_DATA}/mtg.sqlite` read `/mtg.sqlite`. Split from build_spec (the
dataclass contract + parser) to keep that file under the 300-line rule.

Importable on its own: SpecError is resolved lazily at raise time because
build_spec imports this module at its bottom (build_spec is the entry
module of the contract, same seam as spec_rules).
"""

from __future__ import annotations

import os
import re

_BRACED = re.compile(r"\$\{(\w+)\}")


def expand_paths(node, where: str = ""):
    """Return a copy of `node` with every string expanded; `where` is the
    dotted key path carried down the recursion for error messages."""
    if isinstance(node, str):
        return _expand_str(node, where)
    if isinstance(node, dict):
        return {k: expand_paths(v, f"{where}.{k}" if where else str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_paths(v, f"{where}[{i}]") for i, v in enumerate(node)]
    return node


def _expand_str(value: str, where: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        found = os.environ.get(name)
        if not found:
            from forge.build_spec import SpecError

            state = "is empty" if found is not None else "is not set"
            raise SpecError(
                f"{where}: ${{{name}}} {state}; export {name}=/path "
                "(spec paths stay portable, see docs/usage.md section 13)"
            )
        return found

    value = _BRACED.sub(sub, value)
    # today's rule kept: "~/..." instead of a hardcoded home, and the
    # variable itself may hold "~/..." since this runs after substitution.
    return os.path.expanduser(value) if value.startswith("~/") else value
