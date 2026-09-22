# benchmarks

measured 2026-09-10 on arm64 darwin 25.6.0, python 3.12.14, single thread, n=100,000 synthetic clustered l2-normalized rows x 384 dims (2000 centers), 200 queries, k=10, seed 7. reproduce: `.venv/bin/python python/tools/bench_competitors.py --n 100000 --dim 384 --queries 200`.

> [!TIP]
> verify these results on your own hardware with your own parameters: the command above regenerates the whole table, and `--n`, `--dim`, `--queries` set the corpus and the query count. every urna row is a real build, opened and validated before the first query.

| system | path | build (s) | bytes on disk | cold open + 1st query (ms) | p50 (ms) | p99 (ms) | recall@10 | rebuild byte-identical | integrity check |
|---|---|---|---|---|---|---|---|---|---|
| urna (exact) | exact | 2.14 | 165,290,134 | 292.3 | 7.803 | 8.315 | 1.0 | yes | yes (sha256 per section + file + content) |
| urna (hybrid) | ann (hnsw) | 172.82 | 163,221,498 | 356.1 | 0.721 | 1.021 | 1.0 | yes | yes (sha256 per section + file + content) |
| usearch | ann (hnsw) | 109.26 | 168,453,808 | 58.1 | 0.672 | 61.422 | 0.995 | yes | no |
| hnswlib | ann (hnsw) | 83.07 | 168,449,236 | 181.5 | 0.324 | 0.525 | 1.0 | yes | no |
| sqlite-vec | exact | 0.81 | 156,606,464 | 50.0 | 19.762 | 24.836 | 1.0 | yes | structural only (pragma integrity_check) |
| lancedb | exact | 0.25 | 153,799,983 | 612.4 | 16.728 | 19.401 | 1.0 | no | no |

how to read it:

- `cold open + 1st query`: wall time of a fresh interpreter that opens the store and answers one query, minus an interpreter doing nothing (3 runs, min). urna's number is dominated by `open` verifying every section checksum and the footer hash over the whole file before serving anything; the other stores trust their bytes.
- `build (s)`: single-threaded everywhere (hnswlib and usearch are told threads=1); urna's hnsw build is the slow row.
- `p50 / p99`: warm, single-threaded, one query at a time, from python. python call overhead is inside every number.
- `recall@k` is against brute force over the same rows; exact paths are asserted at 1.0.
- `rebuild byte-identical`: two builds from the same rows compared by sha256 over the artefact (a directory is hashed file by file).
- `integrity check`: whether the store can prove its own bytes. urna verifies sha256 per section, per file and over the decoded content on `validate()`.
- the same rows written with raw text and with zstd text share one `content_hash`: `True`. re-encoding never moves a `urna://content_hash/chunk_id` citation; the other stores have no equivalent notion.


## charts

the same table, drawn. every number below is a cell of the table above.

warm p50 vs p99 per store, log scale, bottom-left is fastest and flattest

```mermaid
---
config:
  theme: base
  themeVariables:
    quadrant1Fill: "#161b22"
    quadrant2Fill: "#0d1117"
    quadrant3Fill: "#10221a"
    quadrant4Fill: "#0d1117"
    quadrant1TextFill: "#9198a1"
    quadrant2TextFill: "#9198a1"
    quadrant3TextFill: "#22C55E"
    quadrant4TextFill: "#9198a1"
    quadrantPointFill: "#8B5CF6"
    quadrantPointTextFill: "#c9d1d9"
    quadrantXAxisTextFill: "#9198a1"
    quadrantYAxisTextFill: "#9198a1"
    quadrantTitleFill: "#9198a1"
    quadrantInternalBorderStrokeFill: "#30363d"
    quadrantExternalBorderStrokeFill: "#30363d"
    primaryColor: "#161b22"
    primaryBorderColor: "#30363d"
    lineColor: "#30363d"
---
quadrantChart
    title warm p50 vs p99 per store (log scale)
    x-axis "faster p50" --> "slower p50"
    y-axis "flat tail (p99)" --> "long tail (p99)"
    quadrant-1 "slow and spiky"
    quadrant-2 "fast p50, spiky p99"
    quadrant-3 "fast and flat"
    quadrant-4 "slow and flat"
    "hnswlib": [0.08, 0.08] radius: 5, color: #4285F4
    "urna hybrid (verified)": [0.243, 0.197] radius: 6, color: #22C55E
    "usearch": [0.229, 0.92] radius: 5, color: #FF6F00
    "urna exact (verified)": [0.73, 0.567] radius: 6, color: #22C55E
    "lancedb": [0.886, 0.717] radius: 5, color: #DEA584
    "sqlite-vec": [0.92, 0.76] radius: 5, color: #8E44AD
```

warm p50 per store, lower is better

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      titleColor: "#9198a1"
      xAxisLabelColor: "#9198a1"
      xAxisTitleColor: "#9198a1"
      yAxisLabelColor: "#9198a1"
      yAxisTitleColor: "#9198a1"
      plotColorPalette: "#8B5CF6"
---
xychart-beta
  title "warm p50 (ms), k=10, 100,000 x 384 (lower is better)"
  x-axis ["hnswlib", "usearch", "urna hybrid", "urna exact", "lancedb", "sqlite-vec"]
  y-axis "p50 (ms)" 0 --> 22
  bar [0.324, 0.672, 0.721, 7.803, 16.728, 19.762]
```

warm p99 per store, the tail the p50 hides

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      titleColor: "#9198a1"
      xAxisLabelColor: "#9198a1"
      xAxisTitleColor: "#9198a1"
      yAxisLabelColor: "#9198a1"
      yAxisTitleColor: "#9198a1"
      plotColorPalette: "#22C55E"
---
xychart-beta
  title "warm p99 (ms), k=10, 100,000 x 384 (lower is better)"
  x-axis ["hnswlib", "urna hybrid", "urna exact", "lancedb", "sqlite-vec", "usearch"]
  y-axis "p99 (ms)" 0 --> 70
  bar [0.525, 1.021, 8.315, 19.401, 24.836, 61.422]
```

cold open + first query. urna verifies every section checksum and the footer hash before serving; the other stores trust their bytes

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      titleColor: "#9198a1"
      xAxisLabelColor: "#9198a1"
      xAxisTitleColor: "#9198a1"
      yAxisLabelColor: "#9198a1"
      yAxisTitleColor: "#9198a1"
      plotColorPalette: "#4285F4"
---
xychart-beta
  title "cold open + 1st query (ms), fresh interpreter, min of 3"
  x-axis ["sqlite-vec", "usearch", "hnswlib", "urna exact", "urna hybrid", "lancedb"]
  y-axis "ms" 0 --> 700
  bar [50.0, 58.1, 181.5, 292.3, 356.1, 612.4]
```

single-threaded build time. urna's hnsw build is the slow row, 2.1x hnswlib

```mermaid
---
config:
  theme: base
  themeVariables:
    xyChart:
      backgroundColor: "transparent"
      titleColor: "#9198a1"
      xAxisLabelColor: "#9198a1"
      xAxisTitleColor: "#9198a1"
      yAxisLabelColor: "#9198a1"
      yAxisTitleColor: "#9198a1"
      plotColorPalette: "#FF6F00"
---
xychart-beta
  title "build (s), single thread, 100,000 x 384"
  x-axis ["lancedb", "sqlite-vec", "urna exact", "hnswlib", "usearch", "urna hybrid"]
  y-axis "seconds" 0 --> 200
  bar [0.25, 0.81, 2.14, 83.07, 109.26, 172.82]
```

what urna does not do that some of these do: in-place updates or deletes, metadata filtering, concurrent writers, a query language. it is a build-once, ship-and-query file; the table says nothing about workloads that need those.

versions: {"usearch": "2.26.2", "hnswlib": "0.8.0", "sqlite-vec": "0.1.9", "lancedb": "0.38.0", "numpy": "2.5.2"}
