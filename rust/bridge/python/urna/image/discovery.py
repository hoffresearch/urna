"""Discovery layer for image corpora: what goes in, and in what order.

Separate from the builder because the ordering rules are the load-bearing
part. An item's position in this list becomes its corpus ordinal, its frame
number in the encoded stream, and the `byte_start` its citation resolves
through. If discovery is not deterministic, none of those agree between two
builds of the same directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from urna.embed import visionemb


@dataclass
class Item:
    """One corpus entry, before it becomes a chunk."""

    ordinal: int
    render_path: str  # the file that gets embedded and encoded
    origin: str  # human-facing source, relative to the input root
    label: str = ""
    page: int | None = None

    def canonical_text(self) -> str:
        """The stored, citable text for this entry.

        `cite`, `ask`, and `retrieve` return this and nothing else, so it
        carries the identity of the image rather than only a frame number.
        """
        base = self.origin if self.page is None else f"{self.origin} page {self.page + 1}"
        return f"{base} [{self.label}]" if self.label else base


def render_pdf_pages(input_dir: Path, out_dir: Path, dpi: int = 150) -> list[Item]:
    """Render every page of every pdf, keeping the page number.

    The page number is provenance: a hit on page 240 of a field guide is
    only useful if the corpus can say which page it was.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as err:
        raise RuntimeError("pdf input needs PyMuPDF: uv pip install pymupdf") from err

    items: list[Item] = []
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                out_path = out_dir / f"{pdf_path.stem}_p{page_num:05d}.png"
                doc.load_page(page_num).get_pixmap(dpi=dpi).save(str(out_path))
                items.append(
                    Item(
                        ordinal=len(items),
                        render_path=str(out_path),
                        origin=str(pdf_path.relative_to(input_dir)),
                        page=page_num,
                    )
                )
    return items


def render_page(pdf_path: Path, page: int, out_dir: Path, dpi: int = 150) -> Path:
    """Re-render one pdf page, for query-side use after a build.

    Rendered pages are build-time temporaries, so a pdf corpus would
    otherwise be impossible to evaluate once the build finished. The source
    pdf plus the page number is the durable provenance, and re-rendering
    from it is deterministic.
    """
    import fitz  # PyMuPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}_p{page:05d}.png"
    with fitz.open(str(pdf_path)) as doc:
        doc.load_page(page).get_pixmap(dpi=dpi).save(str(out_path))
    return out_path


def collect_images(input_dir: Path, labels: dict[str, str] | None = None) -> list[Item]:
    labels = labels or {}
    items: list[Item] = []
    for path in visionemb.list_images(input_dir):
        rel = str(path.relative_to(input_dir))
        items.append(
            Item(
                ordinal=len(items),
                render_path=str(path),
                origin=rel,
                # labels may be keyed by relative path or by bare stem, so a
                # csv keyed on image id works with no preprocessing step.
                label=labels.get(rel) or labels.get(path.stem) or "",
            )
        )
    return items


def subsample(items: list[Item], sample: int | None, seed: int) -> list[Item]:
    """Take a seeded subset and RENUMBER it.

    Ordinals must stay dense and zero-based after sampling, because they are
    also the frame numbers of the stream that gets encoded from this list.
    """
    if sample is None or sample >= len(items):
        return items
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(len(items), size=sample, replace=False).tolist())
    return [Item(**{**asdict(items[i]), "ordinal": new}) for new, i in enumerate(picked)]


def load_labels(path: Path | None) -> dict[str, str] | None:
    """Accept a json map or a two-column csv (`image_id,label`)."""
    if path is None:
        return None
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        return {row[0].strip(): row[1].strip() for row in rows[1:] if len(row) >= 2}
    return json.loads(path.read_text())
