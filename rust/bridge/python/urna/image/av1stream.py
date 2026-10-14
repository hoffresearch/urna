"""The av1 stream backend of the image corpus: one (or sharded) mp4 per corpus.

Carved out of `urna/image/orchestra.py`, which keeps the dispatcher
(`build_media`), the per-image backends (control, avif, jxl) and the
resume-path frames iterator. The stream is the only backend with an
ordering permutation, so it lives here; the gop decision (`resolve_keyint`)
sits next to the probe in `gopprober.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from urna.image import mediabase
from urna.image.decframes import decode_frames
from urna.image.encstream import INTER_KEYINT, encode_av1, provenance_sha256
from urna.image.gopprober import probe_gop, resolve_keyint


def _av1_sharded(
    paths,
    output_path,
    dataset_name,
    canvas,
    crf,
    speed,
    keyint,
    pix_fmt,
    shard_size,
    tune="default",
    fps=1,
    contiguous=False,
) -> dict:
    """Consecutive ~`shard_size`-frame segments with an index in the manifest.

    Sharding bounds the worst-case seek walk and is the shape a future
    append will merge into; true append (index merge) is NOT implemented
    and is declared as such. Each segment goes through `encode_av1`, so the
    frame-count and pix_fmt guards hold per segment.

    `keyint=None` means gop=auto resolved PER SEGMENT: one probe per shard.
    A single global probe averages regimes away - with order=cluster the
    near-duplicate runs concentrate in a few segments, and those are exactly
    where inter pays (measured 2026-08-31: -29% on same-artwork reprints)
    while unique-image segments keep O(1) all-intra access.
    """
    media_dir = mediabase.media_dir_for(output_path)
    segments = []
    probes: list[dict] = []
    first: dict | None = None
    for seg_idx, start in enumerate(range(0, len(paths), shard_size)):
        chunk = paths[start : start + shard_size]
        seg_keyint = keyint
        if keyint is None:
            probe = probe_gop(
                chunk,
                canvas,
                crf=crf,
                preset=speed,
                pix_fmt=pix_fmt,
                tune=tune,
                contiguous=contiguous,
            )
            seg_keyint = 1 if probe["decision"] == "intra" else INTER_KEYINT
            probes.append({"segment": seg_idx, **probe})
        name = f"{dataset_name}-av1-{seg_idx:03d}.mp4"
        info = encode_av1(
            chunk,
            media_dir / name,
            canvas=canvas,
            crf=crf,
            preset=speed,
            keyint=seg_keyint,
            pix_fmt=pix_fmt,
            tune=tune,
            fps=fps,
        )
        first = first or info
        segments.append(
            {
                "uri": name,
                "start_frame": start,
                "n_frames": info["frame_count"],
                "output_bytes": info["output_bytes"],
                "media_sha256": info["media_sha256"],
                "keyint": seg_keyint,
            }
        )
    total = sum(s["output_bytes"] for s in segments)
    source_bytes = sum(p.stat().st_size for p in paths)
    seg_keyints = {s["keyint"] for s in segments}
    top_keyint = seg_keyints.pop() if len(seg_keyints) == 1 else None
    toolchain = dict(first["toolchain"])
    toolchain["params"] = {**toolchain["params"], "keyint": top_keyint}
    return {
        "backend": "av1",
        "codec": "libsvtav1",
        "crf": crf,
        "preset": speed,
        "keyint": top_keyint,
        "pix_fmt": first["pix_fmt"],
        "canvas": [canvas[0], canvas[1]],
        "frame_count": sum(s["n_frames"] for s in segments),
        "source_bytes": source_bytes,
        "output_bytes": total,
        "compression_ratio": round(source_bytes / total, 2) if total else 0.0,
        "shard_size": shard_size,
        "segments": segments,
        "gop_probes": probes,
        "toolchain": toolchain,
        "provenance_sha256": provenance_sha256(toolchain),
    }


def build_av1(
    render_paths,
    output_path,
    dataset_name,
    canvas,
    crf,
    speed,
    all_intra,
    pix_fmt,
    gop_policy,
    order,
    shard_size,
    tune="default",
    fps=1,
) -> dict:
    n = len(render_paths)
    order = list(order) if order is not None else list(range(n))
    paths = [render_paths[i] for i in order]
    sharded = bool(shard_size) and n > shard_size
    # an engineered order (cluster/similarity) puts the redundancy between
    # NEIGHBOURS: the probe must sample contiguous windows there, or it
    # erases the very signal the ordering created.
    ordered = any(a != b for a, b in zip(order, range(n), strict=True))
    if sharded and gop_policy == "auto" and not all_intra:
        # per-segment resolution: keyint=None tells _av1_sharded to probe
        # each shard on its own (RFC-2 pendencia 2).
        keyint, gop_record = None, None
    else:
        keyint, gop_record = resolve_keyint(
            paths, canvas, crf, speed, pix_fmt, gop_policy, all_intra, tune, ordered
        )
    media_dir = mediabase.media_dir_for(output_path)
    if sharded:
        media = _av1_sharded(
            paths,
            output_path,
            dataset_name,
            canvas,
            crf,
            speed,
            keyint,
            pix_fmt,
            shard_size,
            tune=tune,
            fps=fps,
            contiguous=ordered,
        )
        probes = media.pop("gop_probes")
        if gop_record is None:
            kinds = {p["decision"] for p in probes}
            gop_record = {
                "policy": "auto",
                "per_segment": True,
                "decision": kinds.pop() if len(kinds) == 1 else "mixed",
                "segments": probes,
            }
        seg_names = [s["uri"] for s in media["segments"]]
    else:
        media_name = f"{dataset_name}-av1.mp4"
        media = encode_av1(
            paths,
            media_dir / media_name,
            canvas=canvas,
            crf=crf,
            preset=speed,
            keyint=keyint,
            pix_fmt=pix_fmt,
            tune=tune,
            fps=fps,
        )
        media["segments"] = [
            {
                "uri": media_name,
                "start_frame": 0,
                "n_frames": media["frame_count"],
                "output_bytes": media["output_bytes"],
                "media_sha256": media["media_sha256"],
            }
        ]
        seg_names = [media_name]
    media["gop"] = gop_record
    if ordered:
        media["order"] = "similarity-greedy"
        media["order_permutation"] = list(order)

    # uris are returned in ITEM order: item i names the stream position the
    # permutation carried it to. vectors/hashes come back in stream order
    # and the caller un-permutes them with `order`.
    sizes = [s["n_frames"] for s in media["segments"]] if "segments" in media else [n]
    bounds = np.cumsum([0, *sizes])
    uris = [""] * n
    for stream_pos, item in enumerate(order):
        seg = int(np.searchsorted(bounds, stream_pos, side="right") - 1)
        uris[item] = f"media://{seg_names[seg]}#frame={stream_pos - int(bounds[seg])}"

    def frames(batch_size: int = 32) -> Iterator[list[np.ndarray]]:
        for name in seg_names:
            yield from decode_frames(media_dir / name, canvas, batch_size=batch_size)

    return {"media": media, "uris": uris, "frames": frames, "order": order}
