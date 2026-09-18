"""urna_forge.py — declarative corpus builds from a TOML/JSON spec (RFC-1).

  python python/tools/urna_forge.py --spec corpus.toml [--sample N] [--models a,b]
      [--out-dir D] [--cache-dir C] [--resume] [--rebuild-only] [--strict-env]
      [--allow-heavy] [--dry-run [--json]]

--dry-run resolves the plan (models, dep status, spaces, outputs) without
loading any model. The rust `urna build` verb is a launcher over this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from forge import model_registry  # noqa: E402
from forge.build_spec import (  # noqa: E402
    SpecError,
    default_model,
    emitted_spaces,
    load_spec,
    validate,
)
from forge.forge_cache import cache_root  # noqa: E402


def dry_run_report(spec, allow_heavy: bool) -> dict:
    models = []
    for ms in spec.models:
        preset = model_registry.PRESETS.get(ms.preset)
        entry: dict = {"preset": ms.preset, "text": ms.text, "image": ms.image, "dims": ms.dims}
        if preset is None:
            entry["status"] = "unknown-preset"
        else:
            entry["kind"] = preset.kind
            entry["executable"] = preset.executable or allow_heavy
            entry["trust_remote_code"] = preset.trust_remote_code
            entry["remote_code_allowed"] = (
                not preset.trust_remote_code or ms.preset in spec.output.allow_remote_code
            )
            model_dir = model_registry.resolve_model_dir(preset, ms.model_path or None)
            entry["model_dir"] = str(model_dir) if model_dir else None
            try:
                model_registry.check_deps(preset)
                entry["deps"] = "ok"
            except model_registry.RegistryError as e:
                entry["deps"] = str(e)
        models.append(entry)
    outputs = []
    if spec.output.mode in ("single", "both"):
        outputs.append(f"{spec.name}.urna")
    if spec.output.mode in ("per-model", "both"):
        dm = default_model(spec)
        outputs.extend(
            f"{spec.name}-{m.preset}.urna"
            for m in spec.models
            # mirror forge_emit: in pure per-model mode the default text model
            # alone would duplicate the single file's core, so emit skips it.
            if not (m.preset == dm.preset and m.image == "none" and spec.output.mode == "per-model")
        )
    return {
        "name": spec.name,
        "source": {"kind": spec.source.kind},
        "image_input_mode": spec.image_input_mode(),
        "media": None
        if spec.media is None
        else {
            "backend": spec.media.backend,
            "crf": spec.media.crf,
            "tune": spec.media.tune,
            "speed": spec.media.speed,
            "order": spec.media.order,
            "dedup": spec.media.dedup,
        },
        "models": models,
        "spaces": [name for _, _, _, name in emitted_spaces(spec)],
        "output_mode": spec.output.mode,
        "outputs": outputs,
        "out_dir": str(Path(spec.output.dir).resolve()),
        "cache_dir": str(cache_root(spec.output.cache_dir).resolve()),
        "provenance": spec.output.provenance,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--sample", type=int)
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="reserved: --sample is evenly spaced and deterministic today, so the "
        "seed has no effect yet; it exists so specs/scripts can pin it ahead of "
        "stochastic sources",
    )
    ap.add_argument("--models", help="comma-separated preset subset")
    ap.add_argument("--out-dir", help="override [output].dir")
    ap.add_argument(
        "--cache-dir",
        help="override [output].cache_dir (else URNA_CACHE_DIR, else "
        "${XDG_CACHE_HOME:-~/.cache}/urna): the shared embed cache root",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rebuild-only", action="store_true")
    ap.add_argument("--strict-env", action="store_true")
    ap.add_argument("--allow-heavy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="dry-run output as JSON")
    args = ap.parse_args()

    try:
        spec = load_spec(args.spec)
        if args.out_dir:
            spec.output.dir = args.out_dir
        if args.cache_dir:
            spec.output.cache_dir = args.cache_dir
        validate(spec, allow_heavy=args.allow_heavy)
        if args.dry_run:
            report = dry_run_report(spec, args.allow_heavy)
            print(json.dumps(report, indent=2) if args.json else _pretty(report))
            return 0
        from forge.forge_pipeline import build

        result = build(
            spec,
            sample=args.sample,
            seed=args.seed,
            models_filter=args.models.split(",") if args.models else None,
            resume=args.resume,
            rebuild_only=args.rebuild_only,
            strict_env=args.strict_env,
            allow_heavy=args.allow_heavy,
        )
        print(json.dumps(result, indent=1))
        return 0
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    except model_registry.RegistryError as e:
        print(f"registry error: {e}", file=sys.stderr)
        return 4


def _pretty(report: dict) -> str:
    lines = [
        f"corpus: {report['name']}  (source={report['source']['kind']}, "
        f"image_input={report['image_input_mode']}, output={report['output_mode']})"
    ]
    if report["media"]:
        m = report["media"]
        lines.append(
            f"media:  {m['backend']} crf={m['crf']} tune={m['tune']} "
            f"speed={m['speed']} order={m['order']} dedup={m['dedup']}"
        )
    for m in report["models"]:
        flags = []
        if not m.get("executable", True):
            flags.append("HEAVY:blocked")
        if m.get("trust_remote_code"):
            flags.append("remote-code:" + ("ok" if m["remote_code_allowed"] else "NOT-ALLOWED"))
        deps = m.get("deps", "?")
        lines.append(
            f"model:  {m['preset']:<20} text={m['text']:<7} image={m['image']:<5} "
            f"dims={m['dims'] or '-'} deps={'ok' if deps == 'ok' else 'MISSING'} "
            f"{' '.join(flags)}"
        )
        if deps != "ok":
            lines.append(f"        -> {deps}")
    lines.append("spaces: " + (", ".join(report["spaces"]) or "(none)"))
    lines.append("files:  " + ", ".join(report["outputs"]))
    lines.append(f"out:    {report['out_dir']}")
    lines.append(f"cache:  {report['cache_dir']}  (embed/<preset>/<triad>.npz, shared)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
