# SPDX-FileCopyrightText: 2026 Brenner Cruvinel, V. Hoff Research
#
# SPDX-License-Identifier: MIT

"""presetqry.py - registry-backed query embedder for the rust CLI.

argv (matching potionqry.py, additively):
  <interp> presetqry.py [--model-path P] [--preset NAME] [--mrl-dim N]
      <manifest_embedding_model> <query>

stdout: one-line JSON {model_hash, fingerprint, embedding_model,
embedding_dim, vector}. The preset resolves from --preset, else by reverse
lookup of the manifest model name; the query is embedded with the preset's
text_query_mode (asymmetric models treat queries and documents differently).
A manifest model no preset names is handed to `searchtxt.py` (the
search-text embedder, beside this module in `embed/`) when
sentence-transformers is importable, so a corpus built with any
sentence-transformers model is askable offline.
--mrl-dim slices+renormalizes the query and reports the truncated dim, for
corpora whose default space was built with mrl_dim.

exit codes: 0 ok, 2 usage, 3 model asset missing, 4 deps/preset problem.
A missing dependency also prints `urna-needs: <pip spec> ...` and weights
missing from the local cache print `urna-fetch: <model>`, one stable line
each on stderr, so the terminal ui can offer to install or fetch them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# run as a file (rust/bridge/python/ or the payload python/), the folder that
# holds urna/ goes first; imported as a module, the package is already there.
if not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


NEEDS = "urna-needs:"


def _has_sentence_transformers() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def _missing_specs(requires: tuple[tuple[str, str], ...]) -> list[str]:
    """Pip specs for every module of `requires` that does not import; each
    fix reads `pip install <spec> [<spec>...]`, quotes optional."""
    import importlib.util

    specs: list[str] = []
    for module, fix in requires:
        if importlib.util.find_spec(module) is None:
            specs.extend(a.strip("\"'") for a in fix.split()[2:])
    return specs


def _st_query(args: argparse.Namespace) -> int:
    """Embed through `rust/bridge/python/urna/embed/searchtxt.py`, the search-text embedder: the
    path of a `st_text` preset (minilm-multilingual) and of any other
    sentence-transformers model no preset names. same encode, same
    fingerprint, offline unless URNA_ALLOW_DOWNLOAD=1, so the model_hash is
    the one the corpus was built with."""
    from urna.embed import searchtxt

    argv = [args.model, args.query]
    if args.model_path:
        argv[:0] = ["--model-path", args.model_path]
    if args.mrl_dim:
        argv[:0] = ["--mrl-dim", str(args.mrl_dim)]
    # searchtxt reports a missing package and a model outside the cache
    # itself, with the same urna-needs / urna-fetch lines.
    return searchtxt.main(argv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path")
    ap.add_argument("--preset")
    ap.add_argument("--mrl-dim", type=int)
    ap.add_argument("model")
    ap.add_argument("query", nargs="?")
    args = ap.parse_args()
    if not args.query:
        print("error: query required", file=sys.stderr)
        return 2

    from urna.model import presetmap as mr

    mr.refuse_connections()
    if args.preset:
        try:
            preset = mr.get_preset(args.preset)
        except mr.RegistryError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4
    else:
        preset = mr.preset_for_embedding_model(args.model)
        if preset is None and _has_sentence_transformers():
            return _st_query(args)
        if preset is None:
            valid = ", ".join(sorted(p.embedding_model for p in mr.PRESETS.values()))
            print(
                f"error: no registry preset embeds '{args.model}'. known manifest "
                f"models: {valid}. pass --preset to force one, or, for a "
                f'sentence-transformers model: pip install "sentence-transformers"',
                file=sys.stderr,
            )
            print(f"{NEEDS} sentence-transformers", file=sys.stderr)
            return 4

    # N11: the manifest of an untrusted .urna must never be enough to run
    # remote model code. The opt-in is explicit and names presets, exactly
    # like [output] allow_remote_code at build time:
    #   URNA_ALLOW_REMOTE_CODE="wemm-2b,jina-v5-omni-nano"
    # The pinned hash allowlist still applies inside create_embedder.
    allowed = frozenset(
        p.strip() for p in os.environ.get("URNA_ALLOW_REMOTE_CODE", "").split(",") if p.strip()
    )
    if preset.trust_remote_code and preset.name not in allowed:
        print(
            f"error: preset '{preset.name}' runs remote model code; opt in with "
            f'URNA_ALLOW_REMOTE_CODE="{preset.name}" (RFC-0 N11)',
            file=sys.stderr,
        )
        return 4
    missing = _missing_specs(preset.requires)
    if missing:
        print(
            f"error: preset '{preset.name}' needs packages that are not installed. "
            f"install with: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        print(f"{NEEDS} {' '.join(missing)}", file=sys.stderr)
        return 4
    if preset.kind == "st_text":
        return _st_query(args)
    try:
        emb = mr.create_embedder(
            preset.name,
            model_path=args.model_path,
            allow_remote_code=allowed,
            allow_heavy=os.environ.get("URNA_ALLOW_HEAVY", "") == "1" or preset.executable,
        )
        vec = emb.embed_texts([args.query], role="query")[0]
    except mr.RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except FileNotFoundError as e:
        print(f"error: model asset missing: {e}", file=sys.stderr)
        if isinstance(e, mr.SnapshotMissing):
            print(f"urna-fetch: {e.repo}", file=sys.stderr)
        return 3

    dim = int(vec.shape[0])
    if args.mrl_dim:
        vec = mr.slice_renorm(vec.reshape(1, -1), args.mrl_dim)[0]
        dim = args.mrl_dim

    json.dump(
        {
            "model_hash": emb.model_hash,
            "fingerprint": emb.fingerprint(),
            "embedding_model": emb.embedding_model,
            "embedding_dim": dim,
            "vector": [float(x) for x in vec],
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
