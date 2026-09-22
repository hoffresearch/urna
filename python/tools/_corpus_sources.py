"""Per-dataset loaders for `urna_build_corpus.py`. Each loader returns
a `pandas.DataFrame` with a fixed shape: `text, label, source, title,
url`. Adding a new dataset means writing one loader and appending it
to `SOURCES` — the pipeline stays the same.

Internal to `python/tools/`. The big script imports this for the
loader registry; nothing else should.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "data" / "demo"

MIN_TEXT_LEN = 20


def _ws(s) -> str:
    return re.sub(r"\s+", " ", s).strip() if isinstance(s, str) else ""


def _row(text, label, source, title="", url=""):
    return {
        "text": _ws(text),
        "label": label,
        "source": source,
        "title": _ws(title),
        "url": _ws(url),
    }


# l── loaders ────────────────────────────────────────────────────────────────


def load_fakebr_hf() -> pd.DataFrame:
    rows = []
    for split in ("train.csv", "test.csv"):
        df = pd.read_csv(DB / "FakeBr-hf" / split)
        df["label"] = df["label"].astype(str).str.strip()
        rows += [_row(r["text"], r["label"], "FakeBr-hf") for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def load_faketrue_br_hf() -> pd.DataFrame:
    rows = []
    for split in ("train.csv", "test.csv"):
        df = pd.read_csv(DB / "FakeTrue.Br-hf" / split)
        df["label"] = df["label"].astype(str).str.strip()
        rows += [_row(r["text"], r["label"], "FakeTrue.Br-hf") for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def load_fake_br_corpus() -> pd.DataFrame:
    df = pd.read_csv(DB / "Fake.br-Corpus" / "preprocessed" / "pre-processed.csv")
    m = {"fake": "False", "true": "True"}
    rows = [
        _row(
            r["preprocessed_news"],
            m.get(str(r["label"]).lower(), str(r["label"])),
            "Fake.br-Corpus",
        )
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


def load_fake_recogna() -> pd.DataFrame:
    df = pd.read_csv(DB / "FakeRecogna" / "dataset" / "FakeRecogna.csv")
    m = {0: "False", 1: "True", "0": "False", "1": "True"}
    rows = []
    for _, r in df.iterrows():
        body = " ".join(
            filter(
                None,
                [
                    _ws(r.get("Titulo", "")),
                    _ws(r.get("Subtitulo", "")),
                    _ws(r.get("Noticia", "")),
                ],
            )
        )
        rows.append(
            _row(
                body,
                m.get(r["Classe"], str(r["Classe"])),
                "FakeRecogna",
                title=r.get("Titulo", ""),
                url=r.get("URL", ""),
            )
        )
    return pd.DataFrame(rows)


def load_faketrue_br() -> pd.DataFrame:
    df = pd.read_csv(DB / "FakeTrue.Br" / "FakeTrueBr_corpus.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            _row(
                r.get("fake", ""),
                "False",
                "FakeTrue.Br",
                title=r.get("title_fake", ""),
                url=r.get("link_f", ""),
            )
        )
        rows.append(_row(r.get("true", ""), "True", "FakeTrue.Br", url=r.get("link_t", "")))
    return pd.DataFrame(rows)


def load_factck_br() -> pd.DataFrame:
    df = pd.read_csv(DB / "factck-br" / "FACTCKBR.tsv", sep="\t")
    rows = []
    for _, r in df.iterrows():
        rating = int(r["ratingValue"]) if pd.notna(r["ratingValue"]) else 3
        if rating == 3:
            continue  # ambiguous
        body = " ".join(
            filter(
                None,
                [
                    _ws(r.get("claimReviewed", "")),
                    _ws(r.get("reviewBody", "")),
                ],
            )
        )
        rows.append(
            _row(
                body,
                "False" if rating <= 2 else "True",
                "factck-br",
                title=r.get("title", ""),
                url=r.get("URL", ""),
            )
        )
    return pd.DataFrame(rows)


def load_bilstm() -> pd.DataFrame:
    df = pd.read_parquet(
        DB / "portuguese-fake-news-classifier-bilstm-combined" / "corpus" / "corpus_test_df.parquet"
    )

    def lbl(v):
        if isinstance(v, bool):
            return "True" if v else "False"
        return str(v).strip()

    rows = [
        _row(r["text"], lbl(r["label"]), "portuguese-fake-news-classifier-bilstm-combined")
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


SOURCES = [
    ("FakeBr-hf", load_fakebr_hf),
    ("FakeTrue.Br-hf", load_faketrue_br_hf),
    ("Fake.br-Corpus", load_fake_br_corpus),
    ("FakeRecogna", load_fake_recogna),
    ("FakeTrue.Br", load_faketrue_br),
    ("factck-br", load_factck_br),
    ("portuguese-fake-news-classifier-bilstm-combined", load_bilstm),
]
