---
project: urna
audience: users
status: active
last-updated: 2026-09-26
domain: demos
---

# quickstart

the smallest corpus that exercises the whole loop: twelve paragraphs of original
prose about urna itself (`docs.jsonl`, one paragraph per row, mirrored from
`demo/corpora/intro/`), the smallest spec that builds them
(`corpus.toml`), and the same build on the python surface (`quickstart.py`).
everything runs offline with the bundled potion embedder: no model download,
no torch, and the file comes out byte-identical on every machine.

## cli

from the repo root (spec paths resolve from the cwd, and `build` needs the
forge under `rust/bridge/python/urna/`):

```
urna build --spec demo/starter/corpus.toml
urna ask demo/starter/out/quickstart.urna "can I use this offline" -k 1
urna retrieve demo/starter/out/quickstart.urna "how do citations work" -k 2 --format jsonl
urna cite demo/starter/out/quickstart.urna 'urna://<content_hash>/<chunk_id>'
urna validate demo/starter/out/quickstart.urna
```

| verb | what it does |
|------|--------------|
| `build` | creates the base: rows + embedding model in, one `.urna` out |
| `ask` | queries it from the terminal: text in, one cited answer out |
| `retrieve` | hands results to another program: json/jsonl of cited spans, `score` is the exact rerank |
| `cite` | resolves a `urna://` citation back to the stored text and its hashes |
| `validate` | proves the file: every checksum, every hash, the manifest contract |

`ask` and `retrieve` print the `citation_id` that `cite` takes. a dev-built
binary is `target/release/urna`; `urna build --spec ... --dry-run` prints the
plan and dependency status without loading anything.

## python

```
pip install "urna[embed]"          # or the dev checkout with rust/bridge/python/urna/_urna.so built
python demo/starter/quickstart.py
```

the script shows the two things the README snippets leave implicit: where the
query vector comes from (`potion_embedder().embed_texts([...])[0]`) and where
`model_hash` comes from (`emb.model_hash()`, the fingerprint the file is built
with and every query is checked against).

## your own corpus

replace `docs.jsonl` with your rows and keep the spec. `source.kind` also
takes `csv`, `sqlite` (a query) and `image_dir` (pdf pages are rendered first
by `rust/bridge/python/urna/entry/imgcorpus.py --pdf`); `[[models]]` takes any
preset in the registry. the full contract, with a worked multi-model spec:
`docs/USAGE.md` section 13.

`out/` is a build artifact and is gitignored.
