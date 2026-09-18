# demo

raw and intermediate data used to build the truw corpus that ships with `urna`. each subdirectory is either a public PT-BR fake-news dataset (kept verbatim from its upstream distribution, license intact) or a derived artifact produced from those sources.

nothing in here is required to use `urna` itself. `urna` reads `.urna` files and only `.urna` files. this directory exists so that the corpus shipped with the project is reproducible: anyone can re-run `python python/tools/urna_build_corpus.py` and get a byte-identical `dat/corpus_next.v1.urna`.

this directory is local-only and gitignored. datasets are fetched on demand from the upstream links below.

## contents

```
dat/demo/
├── FakeBr-hf/                                            7.2k rows  (csv, train+test)
├── FakeTrue.Br-hf/                                       3.6k rows  (csv, train+test)
├── Fake.br-Corpus/                                       7.2k rows  (preprocessed + full_texts)
├── FakeRecogna/                                         11.9k rows  (csv + html report)
├── FakeTrue.Br/                                          3.6k rows  (single csv)
├── factck-br/                                            1.3k rows  (tsv from agência lupa)
├── portuguese-fake-news-classifier-bilstm-combined/      2.2k rows  (parquet, used as test split)
├── corpus-combined/                                       (merge of FakeTrue.Br + Fake.br, v2)
├── corpus-next/embed_cache.sqlite                        (sentence-transformers cache, ~63 MB)
├── truw-built/                                            v2 canonical CSVs + npy embeddings
├── derm/ph2 + derm/ham10000                               dermoscopy images for the image benchmark
├── wsi/CMU-1.svs + wsi/cmu1-tiles                         whole-slide scan and derived tiles
└── pdf/birdcraft-1907.pdf                                 scanned book, public domain
```

each upstream dataset keeps its original license file and structure. see the README inside each subdirectory for citation requirements.

## image corpora

`dat/demo/derm/` holds the dermatology images used to measure the image corpus path. like everything else here it is local-only and gitignored.

```
dat/demo/derm/
├── ph2/images/              200 dermoscopy images + PH2_simple_dataset.csv (diagnosis labels)
└── ham10000/images/         10,015 dermatoscopic images + labels.csv (dx labels), phase 6 used a 2000-sample seed 42
dat/demo/wsi/
├── CMU-1.svs                aperio whole-slide scan (openslide test data)
└── cmu1-tiles/              1210 tiles rendered from CMU-1.svs
dat/demo/pdf/
└── birdcraft-1907.pdf       508-page scanned book, public domain (published 1907)
```

four sources, four media regimes, which is what makes the measurement honest:

- PH2 (Mendonca et al., ADDI project, Universidade do Porto): 200 dermoscopy images labelled Common Nevus, Atypical Nevus, or Melanoma. small, labelled, visually homogeneous. research-use only, check the upstream terms before redistributing.
- HAM10000 (Tschandl et al., Harvard Dataverse, doi:10.7910/DVN/DBW86T): 10,015 dermatoscopic images across seven diagnostic classes. CC BY-NC 4.0, so any derived corpus is non-commercial and must carry attribution.
- CMU-1.svs (openslide test data, Carnegie Mellon): one whole-slide scan tiled into 1210 tiles. freely redistributable test data; the tiles are a derived artifact produced here.
- birdcraft-1907 (Wright, "Birdcraft", 1907): scanned book in the public domain; the pdf pages are the image items.

rebuild the benchmark (the control index is not optional: the compressed numbers mean nothing without it):

```sh
.venv/bin/python python/tools/urna_build_image_corpus.py \
    --input-dir dat/demo/derm/ph2/images --dataset ph2 \
    --output tmp/ph2/ph2.urna --labels dat/demo/derm/ph2/PH2_simple_dataset.csv
.venv/bin/python python/tools/urna_build_image_corpus.py \
    --input-dir dat/demo/derm/ph2/images --dataset ph2 \
    --output tmp/ph2-control/ph2-control.urna --labels dat/demo/derm/ph2/PH2_simple_dataset.csv \
    --control
.venv/bin/python python/tools/urna_image_eval.py \
    --index tmp/ph2/ph2.urna --baseline tmp/ph2-control/ph2-control.urna -k 1 5 10
```

the full variant matrix (av1 crf ladder, avif444, control, dtype rungs, ordering) is one command per dataset with `python/tools/urna_image_sweep.py`; see `doc/usage.md` for the flags and `doc/CHANGELOG` for the measured matrix with confidence intervals.

## offline demo (no downloads)

none of the datasets below are needed for the one-gif demo. `python python/forge/retrieve.py` builds a byte-identical `.urna` from the cc0 demo corpus in `python/forge/demo_corpus` using the vendored potion embedder (numpy + tokenizers, no torch, no network) and prints a cited answer in seconds. it only needs `git lfs pull` to hydrate the potion table.

## upstream sources and download

the folders in this directory are vendored snapshots. if you need to rehydrate them from upstream, use the original sources below.

- `FakeBr-hf` (hugging face): https://huggingface.co/datasets/vzani/corpus-fake-br
- `FakeTrue.Br-hf` (hugging face): https://huggingface.co/datasets/vzani/corpus-faketrue-br
- `corpus-combined` (hugging face): https://huggingface.co/datasets/vzani/corpus-combined
- `portuguese-fake-news-classifier-bilstm-combined` (hugging face): https://huggingface.co/vzani/portuguese-fake-news-classifier-bilstm-combined
- `Fake.br-Corpus` (github): https://github.com/roneysco/Fake.br-Corpus
- `FakeTrue.Br` (github): https://github.com/jpchav98/FakeTrue.Br
- `FakeRecogna` (github): https://github.com/Gabriel-Lino-Garcia/FakeRecogna
- `factck-br` (github mirror): https://github.com/opit-research/factck-br

example pull commands:

```sh
huggingface-cli download vzani/corpus-fake-br --repo-type dataset --local-dir dat/demo/FakeBr-hf
huggingface-cli download vzani/corpus-faketrue-br --repo-type dataset --local-dir dat/demo/FakeTrue.Br-hf
huggingface-cli download vzani/corpus-combined --repo-type dataset --local-dir dat/demo/corpus-combined
huggingface-cli download vzani/portuguese-fake-news-classifier-bilstm-combined --local-dir dat/demo/portuguese-fake-news-classifier-bilstm-combined

git clone https://github.com/roneysco/Fake.br-Corpus dat/demo/Fake.br-Corpus
git clone https://github.com/jpchav98/FakeTrue.Br dat/demo/FakeTrue.Br
git clone https://github.com/Gabriel-Lino-Garcia/FakeRecogna dat/demo/FakeRecogna
git clone https://github.com/opit-research/factck-br dat/demo/factck-br
```

## what truw uses this for

truw is a fact-checking pipeline that scores brazilian-portuguese claims against a curated corpus of labeled news articles. the `.urna` file shipped with the application is rebuilt from these seven sources:

1. read each loader (see `python/tools/_corpus_sources.py`)
2. normalize to a common shape (`text, label, source, title, url`)
3. filter rows shorter than 20 characters
4. deduplicate by sha256 of the text, keep first occurrence
5. embed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim 384, L2-normalized)
6. emit a deterministic `.urna` with the model fingerprint stamped in the manifest

the result is `dat/corpus_next.v1.urna`: 30,725 deduped chunks, 119 MB at the `exact` preset, 34 MB at `tiny`. every chunk keeps a `corpus-next://<source>/<sha256>` provenance URI back to the dataset it came from.

`truw-built/` carries the v2 canonical artifacts for the older fact-check pipeline: `truw_canonical_ptbr_v2.csv` (23.5k articles with claim, label, source, category, date), pre-computed embeddings as `.npy`, and the original `truw_ptbr.urna`.

## how to use

### rebuild the unified corpus

```
python python/tools/urna_build_corpus.py
```

writes `dat/corpus_next.v1.urna`. uses `dat/demo/corpus-next/embed_cache.sqlite` so re-runs skip the embedding step. takes ~10 minutes from cold cache, seconds from warm.

with `reproducible=True` (the default in `BuildConfig` for this script) two operators on different machines produce byte-identical files. `file_hash` and `content_hash` will match.

### load one dataset directly in python

each loader returns a `pandas.DataFrame` with columns `text, label, source, title, url`:

```
import sys; sys.path.insert(0, "python/tools")
from _corpus_sources import load_fakebr_hf, load_factck_br, SOURCES

df = load_fakebr_hf()          # 7.2k rows from FakeBr-hf
df = load_factck_br()          # 1.3k rows from factck-br

# or iterate every source in the registry:
for name, loader in SOURCES:
    print(name, len(loader()))
```

### query the truw v2 corpus directly

`truw-built/truw_canonical_ptbr_v2.csv` is the human-readable canonical form. the `.urna` file equivalent (`truw-built/truw_ptbr.urna`) is opened the same way as any other `.urna`:

```
import sys; sys.path.insert(0, "python")
import urna

db = urna.open("dat/demo/truw-built/truw_ptbr.urna")
hits = db.search(qvec, k=5)
```

## why these seven sources

each one fills a gap the others miss:

- `FakeBr-hf` and `Fake.br-Corpus` are the canonical public PT-BR fake-news baseline (UFG / USP work).
- `FakeTrue.Br` and `FakeTrue.Br-hf` add explicit true/fake pairs for the same claims, useful for label calibration.
- `FakeRecogna` is the largest single corpus (11.9k rows), covers politics + entertainment + health.
- `factck-br` carries 1.3k claims reviewed by agência lupa with their original ratings; high-confidence labels.
- `portuguese-fake-news-classifier-bilstm-combined` provides a held-out test split with a known baseline classifier.

deduplication by sha256 collapses the inevitable overlap (FakeRecogna re-publishes some fake.br articles, FakeTrue.Br shares headlines with FakeBr-hf, etc.) so the unified corpus has no double-counting.

## licenses

each subdirectory inherits its upstream license. some are CC-BY-SA, some are MIT, some are unspecified academic distributions. if you redistribute a built `.urna` derived from this directory, the license of the resulting corpus is the union of all upstream licenses (most restrictive wins). check each `README` inside the subdirectories.

the image sources carry their own terms: PH2 is research-use only (ADDI project, Universidade do Porto), HAM10000 is CC BY-NC 4.0 (Tschandl et al., doi:10.7910/DVN/DBW86T, non-commercial with attribution), CMU-1.svs is freely redistributable openslide test data, and birdcraft-1907 is public domain. a `.urna` derived from ph2 or ham10000 is therefore non-commercial and attribution-carrying at best; do not ship one as a product artifact.

### corpus license bill of materials

The shipped `dat/corpus_next.v1.urna` embeds text derived from all seven sources
below. **Verify each upstream license before redistributing** — several are
research/academic distributions without an explicit redistribution grant, and at
least one is share-alike (CC-BY-SA), which is viral over the derived corpus.

| source | upstream | license (verify at source) | redistribution note |
|--------|----------|----------------------------|---------------------|
| `FakeBr-hf` | huggingface `vzani/corpus-fake-br` | see upstream | derived from Fake.br |
| `FakeTrue.Br-hf` | huggingface `vzani/corpus-faketrue-br` | see upstream | |
| `Fake.br-Corpus` | github `roneysco/Fake.br-Corpus` | academic (UFG/USP) — verify grant | attribution expected |
| `FakeTrue.Br` | github `jpchav98/FakeTrue.Br` | academic — verify grant | |
| `FakeRecogna` | github `Gabriel-Lino-Garcia/FakeRecogna` | academic — verify grant | contains health/political claims about named people |
| `factck-br` | github `opit-research/factck-br` (agência lupa) | see upstream — likely attribution/share-alike | fact-check ratings; attribution required |
| `portuguese-fake-news-classifier-bilstm-combined` | huggingface `vzani/...` | see upstream | held-out test split |

**Obligations for redistributors of a built `.urna`:**

- Honor the **most-restrictive** upstream license (CC-BY-SA attribution +
  share-alike propagates to the whole derived corpus).
- Carry **attribution**: the built file's manifest sets `license` and records
  per-source provenance, but `urna cite` does not yet surface a per-chunk
  license/attribution field — carry the attribution alongside the artifact until
  it does.
- The corpus embeds **personal data about named public figures** (political and
  health claims). See the data governance section of [`doc/SECURITY.md`](../../doc/SECURITY.md#data-governance)
  for the erasure/lawful-basis posture. For anything you ship broadly, prefer the
  CC0 `python/forge/demo_corpus` corpus instead of this mixed-license union.
