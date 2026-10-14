"""Persistent worker process for sentence-transformers multimodal embedders.

Loading two trust_remote_code ST models (jina + wemm) in ONE process
collides in transformers' dynamic-module machinery (measured this session:
jina's custom_st ends up wrapping wemm's weights and every image forward
raises "You must specify exactly one of input_ids or inputs_embeds"; purging
sys.modules is not enough). So each st_multimodal adapter owns a worker
process with exactly one model - no collision, and the model's memory is
returned to the OS when the adapter closes.

Protocol (parent = presetmap._SubprocessSTAdapter):
  argv: --preset NAME [--model-path P] [--batch-size N] [--usage-json J]
  stdin: one task-file path per line; task json:
     {"op": "texts"|"paths"|"arrays", "role": "...", "texts": [...],
      "paths": [...], "frames_npz": "...", "out": "out.npz"}
  stdout: "ok <out.npz>" or "err <message>" per task, flushed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# run as a file (rust/bridge/python/ or the payload python/), the folder that
# holds urna/ goes first; imported as a module, the package is already there.
if not __package__:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--model-path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--usage-json", default="{}")
    args = ap.parse_args()

    import numpy as np

    from urna.embed import stbackend
    from urna.model import presetmap

    presetmap.refuse_connections()
    preset = presetmap.get_preset(args.preset)
    model_dir = presetmap.resolve_model_dir(preset, args.model_path)
    if model_dir is not None and preset.remote_code_hashes:
        presetmap.verify_remote_code(preset, model_dir)
    usage = json.loads(args.usage_json)
    emb = stbackend.STMultimodalEmbedder(
        preset,
        model_dir=model_dir,
        batch_size=args.batch_size,
        dtype=usage.get("model_dtype", ""),
        device=usage.get("device_class")
        if usage.get("device_class") not in (None, "auto")
        else None,
        usage=usage,
    )

    for line in sys.stdin:
        task_path = line.strip()
        if not task_path:
            continue
        try:
            task = json.loads(Path(task_path).read_text())
            op, role = task["op"], task.get("role", "document")
            if op == "texts":
                vecs = emb.embed_texts(task["texts"], role=role)
            elif op == "paths":
                vecs = emb.embed_paths(task["paths"])
            elif op == "arrays":
                with np.load(task["frames_npz"]) as z:
                    frames = [z[k] for k in z.files]
                vecs = emb.embed_arrays(frames)
            else:
                raise ValueError(f"unknown op {op}")
            np.savez(task["out"], vectors=vecs)
            print(f"ok {task['out']}", flush=True)
        except Exception as e:  # noqa: BLE001 - the parent turns this into a typed error
            print(f"err {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
