"""read-side neighbor-context reconstruction for the chunk-to-chunk graph.

the counterpart to dropping chunk_overlap at build time: once the overlap
bytes are gone from chunks_canonical, the surrounding context is rebuilt from
the adjacent chunks the graph_adjacency (0x0C) NEXT_CHUNK edges point at,
costing zero extra stored bytes. the chunk order in a corpus IS the
NEXT_CHUNK order (sequential ordinals), so the context window is the in-corpus
slice around an ordinal. pure, deterministic, no i/o.
"""

from __future__ import annotations

from collections.abc import Sequence


def neighbor_context(
    canonical_texts: Sequence[str],
    ordinal: int,
    *,
    radius: int = 1,
    joiner: str = " ",
) -> str:
    """concatenate chunk `ordinal` with its +/- `radius` NEXT_CHUNK-adjacent
    siblings, rebuilding the context the dropped overlap used to carry."""
    if not 0 <= ordinal < len(canonical_texts):
        raise IndexError(f"ordinal {ordinal} out of range 0..{len(canonical_texts)}")
    lo = max(0, ordinal - radius)
    hi = min(len(canonical_texts), ordinal + radius + 1)
    return joiner.join(canonical_texts[lo:hi])
