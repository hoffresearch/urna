[![Urna: offline-first vector database, Rust and Python](https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-hoff-research-db-iage-thumb-git.png)](https://docs.urna.dev/)

# Urna

Lightweight portable binary vector db that fits in your pocket.

A `.urna` file keeps embeddings, HNSW/BM25 indexes and the search contract together. The Rust runtime memory maps it, checks its hashes and reranks every candidate with exact cosine similarity. The CLI and TUI are intentionally simple: `build`, `ask`, `retrieve`.

Made for those tired of yet another cloud database service.

No plans to become a cloud service or to compete with mature projects like Qdrant or Chroma. The focus is exact cosine scores, byte-for-byte contract validation, compression, portability, and verifiable retrieval without any cloud dependency. Privacy First is our mantra.

Recent experiments with datasets [pt-BR fact-check retrieval](https://github.com/brennercruvinel/fakenews-ptbr-urna-benchmark) · [38k Magic cards compressed with AV1](https://github.com/brennercruvinel/mtg-urna-benchmark)

Docs  [docs.urna.dev](https://docs.urna.dev) 

<img src="https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-tui.svg" alt="urna tui: the ask tab with scored hits and the cited text of the selected one" width="100%">

## Install

```sh
brew install hoffresearch/urna/urna
urna setup
```

Every channel installs the same release binary. `urna setup` then completes the install the same way on each: it downloads the offline embedder, prepares a Python env, checks the install and offers a sample corpus to ask.

<details>
<summary><strong>macOS and Linux</strong>: Homebrew, the one-line installer, nix</summary>

<img src="https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-brew.svg" alt="brew install hoffresearch/urna/urna taps the formula and installs the release binary, then urna --version" width="100%">

```sh
brew install hoffresearch/urna/urna
curl -sSf https://raw.githubusercontent.com/hoffresearch/urna/main/tool/tasks/installer.sh | sh
nix run github:hoffresearch/urna
```

</details>

<details>
<summary><strong>Ubuntu and Debian</strong>: apt from urna.dev, or the PPA</summary>

<img src="https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-apt.svg" alt="the urna.dev/apt keyring, the apt source with signed-by, apt install urna, then urna --version" width="100%">

```sh
curl -fsSL https://urna.dev/apt/urna.gpg | sudo tee /usr/share/keyrings/urna.gpg > /dev/null
echo 'deb [signed-by=/usr/share/keyrings/urna.gpg] https://urna.dev/apt stable main' | sudo tee /etc/apt/sources.list.d/urna.list
sudo apt update && sudo apt install urna
```

On Ubuntu 22.04, 24.04 and 26.04, the PPA builds the same version from source: `sudo add-apt-repository ppa:hoffresearch/urna && sudo apt install urna`. Use one of the two: urna.dev/apt serves the release's own binary, the PPA one that Launchpad compiled; with both configured, apt always takes urna.dev's.

</details>

<details>
<summary><strong>Fedora and RHEL</strong>: dnf from COPR</summary>

<img src="https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-copr.svg" alt="dnf copr enable hoffresearch/urna, dnf install urna, then urna --version, on Fedora 44" width="100%">

```sh
sudo dnf copr enable hoffresearch/urna
sudo dnf install urna
```

Fedora 43 and later, rawhide, and EPEL 9 and 10, on x86_64 and aarch64. The package holds the release's own static binary.

</details>

<details>
<summary><strong>Windows</strong>: scoop, winget, Chocolatey</summary>

```powershell
scoop bucket add hoffresearch https://github.com/hoffresearch/scoop-urna
scoop install urna
winget install HoffResearch.Urna
choco install urna
```

winget and Chocolatey list a version once their moderators approve it.

</details>

<details>
<summary><strong>Language package managers</strong>: npm, cargo, conda, pip</summary>

<img src="https://raw.githubusercontent.com/hoffresearch/urna/main/docs/img/urna-conda.svg" alt="micromamba create -c hoffresearch urna, then urna --version in that environment" width="100%">

```sh
npm install -g @urna/cli
cargo install urna
conda install -c hoffresearch urna
pip install "urna[embed]"
```

npm, cargo and conda install the CLI. pip installs the Python library and a read-only `urna` command (`validate`, `inspect`, `stats`, `search`), with no setup step.

</details>

Binaries for Linux, macOS and Windows are also on the [releases](https://github.com/hoffresearch/urna/releases), each with its `.sha256`. What each channel lays down, how it updates, Docker, `cargo binstall` and how to verify a download are in the [install reference](https://github.com/hoffresearch/urna/blob/main/docs/USAGE.md#reference).

`urna setup` shows the plan before it writes anything and ends on the doctor checks. A corpus built with a heavier model, like the pt-BR MiniLM, needs that model on the machine: `urna setup --model minilm-multilingual` installs it (or `m` on the plan screen), and the ask tab of `urna tui` offers the same install when a query needs it. Nothing is downloaded until you say so.


`urna tui` opens a corpus, validates it, and lets you ask it questions. Each hit shows its score, the stored text and its citation.

```sh
urna tui my_corpus.urna
```

## Quickstart

`urna setup` ends by downloading a sample corpus of about 20 MB: Brazilian Portuguese news articles and fact-checked claims from a [fake-news benchmark](https://github.com/brennercruvinel/fakenews-ptbr-urna-benchmark), true and fake side by side. A hit carries no true or fake label, so an answer can be a fake claim quoted as it was published. It lands in `~/.local/share/urna/sample/` (`$XDG_DATA_HOME/urna/sample/` when that is set, `%LOCALAPPDATA%\urna\sample\` on Windows); `urna setup --no-sample` skips it.

```sh
urna tui ~/.local/share/urna/sample/fakenews.urna
```

```sh
urna ask ~/.local/share/urna/sample/fakenews.urna "imposto sobre pix" -k 2
```

To build a corpus, `demo/starter/` has twelve short paragraphs and the spec that builds them. From a checkout:

```sh
urna build --spec demo/starter/corpus.toml
```

```sh
urna ask demo/starter/out/quickstart.urna "can I use this offline" -k 1
```

```sh
urna retrieve demo/starter/out/quickstart.urna "how do citations work" -k 2 --format jsonl
```

```sh
urna cite demo/starter/out/quickstart.urna 'urna://sha256:b7c81eb1.../sha256:b5dfeb09...'
```

```sh
urna validate demo/starter/out/quickstart.urna
```

`ask` prints the answer with its citation, `retrieve` prints JSON for another program, `cite` turns a citation back into the stored text, and `validate` checks every hash. To build from your own rows, see [usage](https://github.com/hoffresearch/urna/blob/main/docs/USAGE.md).

Doc  [docs.urna.dev](https://docs.urna.dev)

Web [www.urna.dev](https://urna.dev)

----

## License

MIT, see [LICENSE](https://github.com/hoffresearch/urna/blob/main/LICENSE). Author: Brenner Cruvinel.
Made it simple, but significant (∂μfμν = jν)

----
<details>
## What the file guarantees

| Property | How |
|----------|-----|
| Self-contained | The file is the whole database. Copy it like a SQLite file. |
| Verifiable | SHA-256 over the whole file and over the decoded content, plus a checksum on the header and on every section. `urna cite` resolves any citation. |
| Reproducible | Same chunks, same model and same urna version give a byte-identical file on any machine. |
| Offline | The runtime never opens a socket. The CLI refuses a query from another model at the `model_hash` check; in Python, `retrieve` does when you pass `expected_model_hash`, as below. |

## Python

```python
import urna
from urna.embed.potiontab import potion_embedder

emb = potion_embedder()
db = urna.open("my_corpus.urna")
qvec = emb.embed_texts(["can I use this offline"])[0]

hits = db.retrieve(qvec, 5, expected_model_hash=emb.model_hash(), query_text="can I use this offline")
print(hits[0].citation_id, hits[0].score, hits[0].text)
```

<details>
<summary>Search variants, validate, build</summary>

```python
db.search(qvec, 5)                                     # exact
db.search_ann(qvec, 5, 400)                            # hnsw beam (floor: the build's ef_construction), then exact rerank
db.search_hybrid(qvec, "vacina contra covid", 5, 100)  # bm25 + vectors, exact rerank
db.search_graph(qvec, 5, hops=2, ef=100)               # chunk graph from the seeds
db.search_space("clip-vit-b32", ivec, 5)               # one named multimodal space

assert db.validate() is True
info = db.inspect()
```

Each chunk is a dict with `canonical_text`, `source_uri`, `byte_start`, `byte_end` and `embedding`:

```python
urna.build(
    output_path="my_corpus.urna",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dim=384,
    chunker_version="fixed-512/1",
    model_hash=model_hash,
    chunks=chunks,
    reproducible=True,
    preset="hybrid",
)
```

`python demo/starter/quickstart.py` runs the whole loop, build to cited hits, with no network.

</details>

## CLI

The engine verbs take a file and a vector; two of them run Python (`search-text` for its embedder, `doctor` to probe the environment), the other ten never do. The agent verbs (`ask`, `retrieve`, `build`) take text and use the offline embedder; `ask` and `retrieve` search by what the file carries: BM25 plus vectors when the file has a BM25 index, HNSW when it has one, exact otherwise. `setup` and `tui` are the terminal UI. `urna --help` lists all three groups.

<details>
<summary>Agent verbs</summary>

```sh
urna ask my_corpus.urna "can I use this offline" -k 3
```

```sh
urna retrieve my_corpus.urna "can I use this offline" -k 5 --format jsonl
```

```sh
urna build --spec corpus.toml --dry-run
```

`build` reads one TOML: the source (SQLite, CSV, JSONL, an image dir), the media settings, and one or more embedding models from the registry (`potion`, `clip-vit-b32`, `siglip2`, `wemm-2b`, ...). Each model becomes a named vector space in the same file. The full spec is in [usage](https://github.com/hoffresearch/urna/blob/main/docs/USAGE.md).

</details>

<details>
<summary>Search</summary>

```sh
urna search my_corpus.urna "[0.1, 0.2, ...]" -k 10
```

```sh
urna search-ann my_corpus.urna "[0.1, 0.2, ...]" -k 10 --ef 800   # the beam floor is the build's ef_construction (400)
```

```sh
urna search-graph my_corpus.urna "[0.1, 0.2, ...]" -k 10 --hops 2 --ef 100
```

```sh
urna search-space my_corpus.urna "[0.1, ...]" --space "wemm-2b@256" -k 5
```

```sh
urna search-text my_corpus.urna "vacina contra covid funciona" -k 5
```

</details>

<details>
<summary>Inspect, validate, stats, cite, media, benchmark, doctor</summary>

```sh
urna inspect my_corpus.urna --json
```

```sh
urna validate my_corpus.urna
```

```sh
urna stats my_corpus.urna
```

```sh
urna cite my_corpus.urna 'urna://sha256:1aa9.../sha256:8f314...'
```

```sh
urna media my_corpus.urna --export DIR
```

```sh
urna benchmark my_corpus.urna -q 100 -k 10 --ann 100 --madvise-cold
```

```sh
urna doctor
```

</details>

## Benchmarks

100,000 x 384 rows, k=10, one thread, same machine for every store.

| Store | p50 (ms) | p99 (ms) | Cold open (ms) |
|-------|---------:|---------:|---------------:|
| hnswlib | 0.32 | 0.53 | 182 |
| usearch | 0.67 | 61.4 | 58 |
| Urna hybrid | 0.72 | 1.02 | 356 |
| Urna exact | 7.80 | 8.32 | 292 |
| LanceDB | 16.7 | 19.4 | 612 |
| sqlite-vec | 19.8 | 24.8 | 50 |

Both Urna rows return recall@10 = 1.000. Urna's cold open includes checking every section hash before the first answer. Urna does not do updates, filters or concurrent writers. Method and the full table: [docs/BENCH.md](https://github.com/hoffresearch/urna/blob/main/docs/BENCH.md).

<details>
<summary>Presets: size vs recall</summary>

| Preset | Embeddings | Index | Size | Recall@10 |
|--------|------------|-------|-----:|----------:|
| `exact` | float32 | | 1.000 | 1.000 |
| `compressed` | float16 | | 0.339 | 1.000 |
| `tiny` | int8 | HNSW | 0.256 | 0.992 |
| `micro` | mrl256-int8 | HNSW | 0.223 | 0.810 |
| `nano` | int4 | HNSW | 0.209 | 0.913 |
| `hybrid` | float32 | HNSW + BM25 | 0.609 | 1.000 |

Measured on a 30,725-chunk pt-BR corpus. Recall here is rank stability under quantization, not real-query quality. Details in [usage section 6](https://github.com/hoffresearch/urna/blob/main/docs/USAGE.md).

Real-query quality is in [fakenews-ptbr-urna-benchmark](https://github.com/brennercruvinel/fakenews-ptbr-urna-benchmark): seven public pt-BR fake-news datasets deduplicated into 23,335 documents, 2,601 queries with relevance judgments, three embedders and three presets each, rebuildable from pinned sources. On the `exact` preset, nDCG@10 is 0.528 for mpnet, 0.503 for the multilingual MiniLM and 0.326 for potion. The files are on [Hugging Face](https://huggingface.co/datasets/brennercruvinel/fakenews-ptbr-urna-benchmark).

</details>

<details>
<summary>Images: 38,627 Magic cards in one file</summary>

| Profile | Media | File | Vs the JPEG source |
|---------|-------|-----:|-------------------:|
| `archive` | JPEG XL, byte-reversible | 3.61 GB | 1.10x |
| `stills` | AVIF, one image per file (q48) | 1.20 GB | 3.32x |
| `stills-av1` | AV1 all-intra crf35 | 1.37 GB | 2.89x |
| `retrieval` | AV1 all-intra crf50 | 533 MB | 7.46x |

The profile names are the forge's (`[media] profile = "..."`, usage section 14). mtg-urna-benchmark names its v0.3 files from before AVIF became the forge's `stills`: its `stills` file is this table's `stills-av1` row, and the AVIF q48 row is its candidate `v03-avif-q48`. Text-to-image hit@1 over every card: SigLIP2 0.750, wemm-2b 0.744, Jina 0.336, CLIP 0.098. The benchmark is [mtg-urna-benchmark](https://github.com/brennercruvinel/mtg-urna-benchmark), and the `.urna` files are on [Hugging Face](https://huggingface.co/datasets/brennercruvinel/mtg-urna-benchmark); the recipes the forge uses are recorded in `rust/bridge/python/urna/image/mediaplan.py` and `docs/CHANGELOG`.

</details>

## Cite

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23371900.svg)](https://doi.org/10.5281/zenodo.23371900) [![Software Heritage](https://archive.softwareheritage.org/badge/origin/https://github.com/hoffresearch/urna/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/hoffresearch/urna)

Cite the version you used: its release tag holds the `CITATION.cff` with that version's metadata (`https://github.com/hoffresearch/urna/blob/v<version>/CITATION.cff`), and GitHub's "Cite this repository" renders the [latest one](https://github.com/hoffresearch/urna/blob/main/CITATION.cff). Every release tag is archived in Software Heritage, outside GitHub; its identifier is `swh:1:rel:` followed by the tag's git object id. Zenodo archives every release too: the DOI [10.5281/zenodo.23371900](https://doi.org/10.5281/zenodo.23371900) always resolves to the latest version, and each version has a DOI of its own on its Zenodo record.

## Reference

- [docs.urna.dev](https://docs.urna.dev): the documentation site, with guides, concepts and the full reference
- [docs/USAGE.md](https://github.com/hoffresearch/urna/blob/main/docs/USAGE.md): every verb, presets, models, builds, install channels
- [docs/BENCH.md](https://github.com/hoffresearch/urna/blob/main/docs/BENCH.md): how the numbers were measured
- [fakenews-ptbr-urna-benchmark](https://github.com/brennercruvinel/fakenews-ptbr-urna-benchmark) and [mtg-urna-benchmark](https://github.com/brennercruvinel/mtg-urna-benchmark): the text and image benchmarks, with their files on Hugging Face
- [docs/SECURITY.md](https://github.com/hoffresearch/urna/blob/main/docs/SECURITY.md): reporting, hardening, data governance
- [docs/CHANGELOG](https://github.com/hoffresearch/urna/blob/main/docs/CHANGELOG): releases with measured numbers
- [docs/ATLAS.toml](https://github.com/hoffresearch/urna/blob/main/docs/ATLAS.toml): the architecture map
- [docs/adr](https://github.com/hoffresearch/urna/blob/main/docs/adr/README.md): architecture decision records, by category

The crates are `urna-format` (the container), `urna-engine` (search), `urna` (the binary) and `urna-bridge` (the PyO3 bridge behind the Python package).

> Renamed from `nest` after 0.4.0. A `.nest` file written by 0.4.0 still opens.

[Hoff Research](https://hoffresearch.com)

</details>
