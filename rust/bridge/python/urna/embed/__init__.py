"""embed: the embedders of the ingestion layer and the query scripts.

the default is the REAL semantic static table (model2vec/potion-base-8M,
offline, no torch); the #04 lexical bag-of-words stays available as the
zero-dependency floor (lexical_embedder). both reuse the one authoritative
chunker (urna.pipes.buildfile.chunk_text).

the re-exports load on first use: potiontab forces the hub offline when it
is imported, and the query scripts of this package (searchtxt.py with its
URNA_ALLOW_DOWNLOAD=1 opt-in) must not inherit that by importing the package.
"""

import importlib

_EXPORTS = {
    "StaticEmbedder": ("lexifloor", "StaticEmbedder"),
    "lexical_embedder": ("lexifloor", "default_embedder"),
    "PotionEmbedder": ("potiontab", "PotionEmbedder"),
    "default_embedder": ("potiontab", "default_embedder"),
    "potion_embedder": ("potiontab", "potion_embedder"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module, attr = _EXPORTS[name]
        value = getattr(importlib.import_module(f"{__name__}.{module}"), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StaticEmbedder",
    "PotionEmbedder",
    "default_embedder",
    "potion_embedder",
    "lexical_embedder",
]
