# what forge does

forge is the ingestion layer. it turns messy, heterogeneous inputs such as
pdfs, plain text, datasets, and archives into a deterministic canonical
intermediate that the urna build path already understands.

forge lives outside the sovereign core so its heavier dependencies never touch
the small, frozen format and runtime crates. the rule that keeps builds
reproducible is simple: the determinism anchor is the canonical extracted text
plus the embedding model fingerprint, not the raw input. the same canonical
text and the same fingerprint always produce a byte-identical file, even if
the original pdf was re-saved or the extraction engine changed.

forge does not invent a second chunker. there is one authoritative chunker,
and forge calls it, so chunk identities never drift between tools.
