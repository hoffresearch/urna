"""adapters for bench_competitors.py: one class per system, same four verbs.

every adapter builds from the SAME float32 rows, persists to `path`, reports
the bytes on disk, answers `search(q, k) -> list[int]` (row indices), and can
reopen itself in a fresh process for the cold-open measurement. adapters
never hide a capability they lack: `ann` is None when the system has no
approximate path here, `byte_identical_rebuild` is measured, not assumed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def dir_or_file_bytes(path: str) -> int:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


def sha256_tree(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()


def rm(path: str) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    elif p.exists():
        p.unlink()


class UrnaSystem:
    """urna via the python binding; preset selects the stored dtype / index."""

    name = "urna"

    def __init__(self, preset: str, ann: bool, text_encoding: str | None = None):
        import urna

        self.urna = urna
        self.preset = preset
        self.text_encoding = text_encoding
        self.has_ann = ann
        self.name = f"urna ({preset})"
        self.db = None
        self.order: dict[str, int] = {}

    def build(self, rows: np.ndarray, path: str) -> None:
        rm(path)
        chunks = [
            {
                "canonical_text": f"row {i}",
                "source_uri": "synthetic",
                "byte_start": i * 8,
                "byte_end": i * 8 + 7,
                "embedding": rows[i].tolist(),
            }
            for i in range(rows.shape[0])
        ]
        self.urna.build(
            output_path=path,
            embedding_model="synthetic",
            embedding_dim=int(rows.shape[1]),
            chunker_version="bench/1",
            model_hash="sha256:" + "0" * 64,
            chunks=chunks,
            reproducible=True,
            preset=self.preset,
            text_encoding=self.text_encoding,
            hnsw_m=16,
            hnsw_ef_construction=200,
        )

    def open(self, path: str) -> None:
        self.db = self.urna.open(path)
        # chunk ids are content-addressed; file order == insertion order, so
        # the position of a chunk id IS the row index.
        self.order = {cid: i for i, cid in enumerate(self.db.chunk_ids())}

    def search(self, q: np.ndarray, k: int) -> list[int]:
        qv = q.tolist()
        hits = self.db.search_ann(qv, k, 100) if self.has_ann else self.db.search(qv, k)
        return [self.order[h.chunk_id] for h in hits]

    def validate(self) -> str:
        return "yes (sha256 per section + file + content)" if self.db.validate() else "FAILED"

    def content_hash(self) -> str:
        return self.db.content_hash

    reopen_snippet = (
        "import sys; sys.path.insert(0, 'python'); import urna, json; "
        "db = urna.open(PATH); db.search(Q, 10)"
    )


class UsearchSystem:
    name = "usearch"

    def __init__(self):
        from usearch.index import Index

        self.Index = Index
        self.has_ann = True
        self.index = None

    def build(self, rows: np.ndarray, path: str) -> None:
        rm(path)
        # connectivity 16 / expansion_add 200 / expansion_search 100: the same
        # hnsw knobs urna and hnswlib run with in this table.
        idx = self.Index(
            ndim=rows.shape[1],
            metric="cos",
            dtype="f32",
            connectivity=16,
            expansion_add=200,
            expansion_search=100,
        )
        idx.add(np.arange(rows.shape[0], dtype=np.uint64), rows, threads=1)
        idx.save(path)

    def open(self, path: str) -> None:
        self.index = self.Index.restore(path, view=True)
        self.index.expansion_search = 100

    def search(self, q: np.ndarray, k: int) -> list[int]:
        return [int(x) for x in self.index.search(q, k).keys]

    def validate(self) -> str:
        return "no"

    reopen_snippet = (
        "from usearch.index import Index; import numpy as np; "
        "ix = Index.restore(PATH, view=True); ix.expansion_search = 100; "
        "ix.search(np.array(Q, dtype='float32'), 10)"
    )


class HnswlibSystem:
    name = "hnswlib"

    def __init__(self):
        import hnswlib

        self.hnswlib = hnswlib
        self.has_ann = True
        self.index = None
        self.dim = 0

    def build(self, rows: np.ndarray, path: str) -> None:
        rm(path)
        self.dim = rows.shape[1]
        idx = self.hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(max_elements=rows.shape[0], ef_construction=200, M=16, random_seed=42)
        idx.add_items(rows, np.arange(rows.shape[0]), num_threads=1)
        idx.save_index(path)

    def open(self, path: str) -> None:
        self.index = self.hnswlib.Index(space="cosine", dim=self.dim)
        self.index.load_index(path)
        self.index.set_ef(100)

    def search(self, q: np.ndarray, k: int) -> list[int]:
        labels, _ = self.index.knn_query(q.reshape(1, -1), k=k)
        return [int(x) for x in labels[0]]

    def validate(self) -> str:
        return "no"

    reopen_snippet = (
        "import hnswlib, numpy as np; ix = hnswlib.Index(space='cosine', dim=DIM); "
        "ix.load_index(PATH); ix.set_ef(100); ix.knn_query(np.array([Q], dtype='float32'), k=10)"
    )


class SqliteVecSystem:
    name = "sqlite-vec"

    def __init__(self):
        import sqlite3

        import sqlite_vec

        self.sqlite3 = sqlite3
        self.sqlite_vec = sqlite_vec
        self.has_ann = False
        self.conn = None

    def _connect(self, path: str):
        conn = self.sqlite3.connect(path)
        conn.enable_load_extension(True)
        self.sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def build(self, rows: np.ndarray, path: str) -> None:
        rm(path)
        conn = self._connect(path)
        conn.execute(f"create virtual table v using vec0(embedding float[{rows.shape[1]}])")
        with conn:
            conn.executemany(
                "insert into v(rowid, embedding) values (?, ?)",
                ((i, rows[i].astype(np.float32).tobytes()) for i in range(rows.shape[0])),
            )
        conn.close()

    def open(self, path: str) -> None:
        self.conn = self._connect(path)

    def search(self, q: np.ndarray, k: int) -> list[int]:
        cur = self.conn.execute(
            "select rowid from v where embedding match ? order by distance limit ?",
            (q.astype(np.float32).tobytes(), k),
        )
        return [int(r[0]) for r in cur.fetchall()]

    def validate(self) -> str:
        return "structural only (pragma integrity_check)"

    reopen_snippet = (
        "import sqlite3, sqlite_vec, numpy as np; c = sqlite3.connect(PATH); "
        "c.enable_load_extension(True); sqlite_vec.load(c); "
        "c.execute('select rowid from v where embedding match ? order by distance limit 10', "
        "(np.array(Q, dtype='float32').tobytes(),)).fetchall()"
    )


class LanceDbSystem:
    name = "lancedb"

    def __init__(self):
        import lancedb
        import pyarrow as pa

        self.lancedb = lancedb
        self.pa = pa
        self.has_ann = False
        self.table = None

    def build(self, rows: np.ndarray, path: str) -> None:
        rm(path)
        db = self.lancedb.connect(path)
        arr = self.pa.FixedSizeListArray.from_arrays(
            self.pa.array(rows.astype(np.float32).ravel()), rows.shape[1]
        )
        tbl = self.pa.table({"id": self.pa.array(range(rows.shape[0])), "vector": arr})
        db.create_table("v", tbl)

    def open(self, path: str) -> None:
        self.table = self.lancedb.connect(path).open_table("v")

    def search(self, q: np.ndarray, k: int) -> list[int]:
        res = self.table.search(q.astype(np.float32)).metric("cosine").limit(k).to_list()
        return [int(r["id"]) for r in res]

    def validate(self) -> str:
        return "no"

    reopen_snippet = (
        "import lancedb, numpy as np; t = lancedb.connect(PATH).open_table('v'); "
        "t.search(np.array(Q, dtype='float32')).metric('cosine').limit(10).to_list()"
    )


def cold_open_ms(snippet: str, path: str, q: list[float], dim: int, python: str) -> float:
    """wall time of a fresh interpreter that opens the store and answers one
    query, minus the same interpreter doing nothing (python startup)."""
    code = snippet.replace("PATH", repr(path)).replace("Q", repr(q)).replace("DIM", str(dim))
    import subprocess

    def run(c: str) -> float:
        t0 = time.perf_counter()
        r = subprocess.run([python, "-c", c], cwd=os.getcwd(), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cold-open snippet failed:\n{c[:200]}...\n{r.stderr[-800:]}")
        return (time.perf_counter() - t0) * 1e3

    base = min(run("pass") for _ in range(3))
    return min(run(code) for _ in range(3)) - base


PYTHON = sys.executable
