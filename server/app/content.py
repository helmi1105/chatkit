# app/content.py
"""The doctrine as the tutor sees it: per-page text blocks used for DIRECT
grounding (injected in the prompt), page images for display, and a small
BM25 index for free questions.

Two sources, best first:
  1. doctrine_pages.json -- PDF embedded text plus conservative visual notes.
     Per-page verification metadata records the audit scope; assistant review
     is not trainer approval. Complex diagrams still require the original PDF.
  2. doctrine_chunks.json -- Tesseract OCR precomputed at Docker build
     (build_doctrine_index.py), text only.

The memento is 24 pages / ~17k characters of OCR, so a KC's page(s) plus its
neighbours fit in one prompt: no tool calls, no retrieval flakiness, and the
generation is one round trip instead of the 3-minute tool loops measured on
the free Mistral tier (cahier 2026-09-15)."""
from __future__ import annotations

import json
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
PAGES_JSON = APP_DIR / os.getenv("DOCTRINE_PAGES_JSON", "doctrine_pages.json")
CHUNKS_JSON = APP_DIR / os.getenv("DOCTRINE_CHUNKS_CACHE", "doctrine_chunks.json")
PAGE_IMAGE_DIR = APP_DIR / "pages"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DOCTRINE_PDF_NAME = os.getenv("DOCTRINE_PDF_NAME", "Charte graphique 2025 - Impression.pdf")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _format_structured_page(item: Dict[str, Any]) -> str:
    lines: List[str] = []
    title = str(item.get("title") or "").strip()
    if title:
        lines.append(f"Titre : {title}")
    for sec in item.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        heading = str(sec.get("heading") or "").strip()
        text = str(sec.get("text") or "").strip()
        if heading and text:
            lines.append(f"{heading} : {text}")
        elif text:
            lines.append(text)
    symbols = [s for s in (item.get("symbols") or []) if isinstance(s, dict)]
    if symbols:
        lines.append("Symboles de la page (forme, couleur, style : signification) :")
        for s in symbols:
            parts = [str(s.get("shape") or "").strip(), str(s.get("color") or "").strip(), str(s.get("style") or "").strip()]
            desc = ", ".join(p for p in parts if p)
            label = str(s.get("label") or "").strip()
            inside = str(s.get("text_inside") or "").strip()
            meaning = str(s.get("meaning") or "").strip()
            extra = f" [{label}]" if label else ""
            extra += f" (inscription : {inside})" if inside else ""
            lines.append(f"- {desc}{extra} : {meaning}")
    for tb in item.get("tables") or []:
        if not isinstance(tb, dict):
            continue
        ttitle = str(tb.get("title") or "Tableau").strip()
        cols = [str(c) for c in (tb.get("columns") or [])]
        lines.append(f"Tableau « {ttitle} »" + (f" ({' | '.join(cols)})" if cols else "") + " :")
        for row in tb.get("rows") or []:
            if isinstance(row, list):
                lines.append("  - " + " | ".join(str(c) for c in row))
        for entry in tb.get("entries") or []:
            if isinstance(entry, dict):
                lines.append("  - " + " | ".join(
                    f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                    for key, value in entry.items()))
            elif isinstance(entry, (str, list)):
                lines.append("  - " + (entry if isinstance(entry, str) else " | ".join(map(str, entry))))
    rules = [str(r).strip() for r in (item.get("rules") or []) if str(r).strip()]
    if rules:
        lines.append("Règles énoncées :")
        lines.extend(f"- {r}" for r in rules)
    return "\n".join(lines).strip()


class Doctrine:
    def __init__(self) -> None:
        self.pages: Dict[int, str] = {}
        self.structured: Dict[int, Dict[str, Any]] = {}
        self.source = "none"
        structured = _load_json(PAGES_JSON)
        if isinstance(structured, dict):
            structured = structured.get("pages")
        if isinstance(structured, list) and structured:
            for item in structured:
                if isinstance(item, dict) and item.get("page"):
                    p = int(item["page"])
                    self.structured[p] = item
                    self.pages[p] = _format_structured_page(item)
            self.source = "vision"
        chunks = _load_json(CHUNKS_JSON)
        if isinstance(chunks, list):
            ocr: Dict[int, List[str]] = {}
            for c in chunks:
                if isinstance(c, dict) and c.get("page"):
                    ocr.setdefault(int(c["page"]), []).append(str(c.get("text") or ""))
            for p, texts in ocr.items():
                text = "\n".join(t for t in texts if t.strip())
                if p not in self.pages:
                    self.pages[p] = text
                    if self.source == "none":
                        self.source = "ocr"
                elif self.source == "vision" and not self.structured.get(p, {}).get("verification"):
                    # keep the OCR running text as a complement (names the
                    # vision model may have paraphrased)
                    self.pages[p] = self.pages[p] + "\nTexte OCR de la page : " + re.sub(r"\s+", " ", text)
        self.revision = hashlib.sha256(json.dumps(self.pages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        self._bm25 = None
        self._page_order: List[int] = sorted(self.pages)
        try:
            from rank_bm25 import BM25Okapi

            docs = [self._tokenize(self.pages[p]) for p in self._page_order]
            self._bm25 = BM25Okapi(docs) if docs else None
        except Exception:
            self._bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zà-ÿœæ0-9]+", text.lower())

    # ---- text access -----------------------------------------------------
    def page_text(self, page: int) -> str:
        return self.pages.get(int(page), "")

    def pages_text(self, pages: List[int], *, max_chars: int = 9000) -> str:
        blocks: List[str] = []
        total = 0
        for p in sorted({int(x) for x in pages if x}):
            text = self.page_text(p)
            if not text:
                continue
            block = f"=== Page {p} ===\n{text}"
            if total + len(block) > max_chars:
                block = block[: max(0, max_chars - total)]
            blocks.append(block)
            total += len(block)
            if total >= max_chars:
                break
        return "\n\n".join(blocks)

    def full_text(self, *, max_chars: int = 40000) -> str:
        return self.pages_text(self._page_order, max_chars=max_chars)

    def rules_for_pages(self, pages: List[int]) -> List[str]:
        """Testable rules the vision transcription extracted for these pages
        (empty when only the OCR is available)."""
        out: List[str] = []
        for p in sorted({int(x) for x in pages if x}):
            item = self.structured.get(p)
            if not item:
                continue
            for r in item.get("rules") or []:
                r = re.sub(r"\s+", " ", str(r)).strip()
                if r and r not in out:
                    out.append(r)
            for s in item.get("symbols") or []:
                if not isinstance(s, dict):
                    continue
                meaning = str(s.get("meaning") or "").strip()
                shape = str(s.get("shape") or "").strip()
                color = str(s.get("color") or "").strip()
                if meaning and (shape or color) and meaning.lower() != "illisible":
                    rule = f"{shape} {color}".strip() + f" = {meaning}"
                    if rule not in out:
                        out.append(rule)
        return out

    def search_pages(self, query: str, k: int = 3) -> List[int]:
        if not self._bm25 or not self._page_order:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._page_order[i] for i in ranked[:k] if scores[i] > 0]

    # ---- assets ----------------------------------------------------------
    @staticmethod
    def page_image_url(page: int) -> Optional[str]:
        path = PAGE_IMAGE_DIR / f"page-{int(page)}.png"
        if not path.exists():
            return None
        return f"{PUBLIC_BASE_URL}/static/pages/page-{int(page)}.png"

    @staticmethod
    def pdf_url(page: int) -> str:
        from urllib.parse import quote

        return f"{PUBLIC_BASE_URL}/static/{quote(DOCTRINE_PDF_NAME)}#page={int(page)}"


@lru_cache(maxsize=1)
def doctrine() -> Doctrine:
    return Doctrine()
