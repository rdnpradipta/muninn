#!/usr/bin/env python3
"""Muninn ingest: PDFs -> page-level chunks -> bge-small embeddings -> Qdrant.

Usage:
    uv run ingest.py                      # local embedded DB (./qdrant_db)
    QDRANT_URL=... QDRANT_API_KEY=... uv run ingest.py   # Qdrant Cloud

Idempotent per document: re-running deletes and re-inserts a doc's points.
"""
import hashlib
import os
import re
import sys
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

import citations  # DOI -> APA (Crossref/DataCite via doi.org, cached)

# --- Config (must match muninn_mcp.py) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384
COLLECTION = "muninn"
CHUNK_SIZE = 1000       # chars
CHUNK_OVERLAP = 150     # chars
PDF_DIR = Path(os.environ.get("PDF_DIR", Path(__file__).parent / "pdfs"))


def get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(Path(__file__).parent / "qdrant_db"))


def doc_id_for(path: Path) -> str:
    """Stable short id from filename."""
    return hashlib.sha1(path.name.encode()).hexdigest()[:12]


def clean_text(t: str) -> str:
    t = t.replace("­", "")            # soft hyphens
    t = re.sub(r"-\n(?=[a-z])", "", t)     # de-hyphenate line breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def guess_title(doc: fitz.Document, path: Path) -> str:
    meta = (doc.metadata or {}).get("title") or ""
    if len(meta.strip()) > 8 and not meta.lower().startswith(("untitled", "microsoft")):
        return meta.strip()
    # fallback: filename after first underscore, minus extension
    name = path.stem
    if "_" in name:
        name = name.split("_", 1)[1]
    return name.replace("-", " ").strip() or path.stem


def chunk_page(text: str) -> list[tuple[str, int, int]]:
    """Split one page's text into overlapping chunks. Never crosses pages.
    Returns list of (chunk_text, char_start, char_end)."""
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [(text, 0, len(text))]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # prefer to break at sentence/paragraph boundary in last 200 chars
            window = text[max(start, end - 200):end]
            m = list(re.finditer(r"[.!?]\s|\n\n", window))
            if m:
                end = max(start, end - 200) + m[-1].end()
        chunks.append((text[start:end].strip(), start, end))
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return [c for c in chunks if len(c[0]) > 40]


def ingest_notes(client: QdrantClient, embedder: TextEmbedding) -> None:
    """Ingest Claude's reading notes from notes/*.jsonl as type=claude_note."""
    import json
    notes_dir = Path(__file__).parent / "notes"
    files = sorted(notes_dir.glob("*.jsonl"))
    if not files:
        print("[notes] none found, skipping")
        return

    # Notes are ingested after the PDFs, so enrich them from each document's
    # meta point. This keeps note search results consistent with raw hits:
    # both carry the DOI and APA-7 in-text citation.
    citations_by_doc: dict[str, dict] = {}
    offset = None
    while True:
        meta_points, offset = client.scroll(
            COLLECTION, limit=500, offset=offset, with_payload=True,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="type", match=models.MatchValue(value="meta"))]),
        )
        for point in meta_points:
            payload = point.payload
            citations_by_doc[payload["doc_id"]] = {
                "doi": payload.get("doi"),
                "apa_inline": payload.get("apa_inline"),
            }
        if offset is None:
            break

    for f in files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            n = json.loads(line)
            did = n["doc_id"]
            client.delete(
                COLLECTION,
                points_selector=models.FilterSelector(filter=models.Filter(must=[
                    models.FieldCondition(key="doc_id",
                                          match=models.MatchValue(value=did)),
                    models.FieldCondition(key="type",
                                          match=models.MatchValue(value="claude_note")),
                ])),
            )
            # chunk the note (no page concept; cite the paper via key_quotes)
            chunks = chunk_page(n["note"]) or [(n["note"], 0, len(n["note"]))]
            texts = [c[0] for c in chunks]
            vectors = list(embedder.embed(texts))
            points = []
            for ci, ((ctext, _, _), vec) in enumerate(zip(chunks, vectors)):
                payload = {"doc_id": did, "title": n["title"],
                           "source_file": n["source_file"], "type": "claude_note",
                           "chunk_index": ci, "text": ctext,
                           **citations_by_doc.get(did, {})}
                if ci == 0 and n.get("key_quotes"):
                    payload["key_quotes"] = n["key_quotes"]
                points.append(models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{did}:note:c{ci}")),
                    vector=vec.tolist(), payload=payload))
            client.upsert(COLLECTION, points=points)
            print(f"[note] {n['title'][:60]!r}  chunks={len(points)}")


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_DIM, distance=models.Distance.COSINE
            ),
        )
        client.create_payload_index(COLLECTION, "doc_id",
                                    models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "type",
                                    models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "page",
                                    models.PayloadSchemaType.INTEGER)
        client.create_payload_index(COLLECTION, "doi",
                                    models.PayloadSchemaType.KEYWORD)

    embedder = TextEmbedding(EMBED_MODEL)
    cite_cache = citations._load_cache()   # shared across docs, saved once below

    for path in pdfs:
        did = doc_id_for(path)
        # idempotent: wipe existing raw/page/meta points for this doc
        client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="doc_id",
                                      match=models.MatchValue(value=did)),
                models.FieldCondition(key="type",
                                      match=models.MatchAny(
                                          any=["raw", "page", "meta"])),
            ])),
        )

        # --- APA citation from the DOI encoded in the filename ---
        cite = citations.citation_for(path.name, cache=cite_cache) or {}
        cite_fields = {"doi": cite.get("doi"),
                       "apa_inline": cite.get("inline")}

        doc = fitz.open(path)
        title = guess_title(doc, path)
        total_pages = doc.page_count
        points = []
        n_chunks = 0
        for pno in range(doc.page_count):
            page_text = clean_text(doc[pno].get_text("text"))
            if not page_text:
                continue
            base = {"doc_id": did, "title": title, "source_file": path.name,
                    "page": pno + 1, "total_pages": doc.page_count,
                    **cite_fields}
            # full page text (type=page) for get_page — zero vector cost is
            # avoided by embedding page text too (cheap, and page-level recall helps)
            for ci, (ctext, cs, ce) in enumerate(chunk_page(page_text)):
                points.append(("raw", f"{did}:p{pno+1}:c{ci}",
                               {**base, "type": "raw", "chunk_index": ci,
                                "char_start": cs, "char_end": ce, "text": ctext}))
                n_chunks += 1
            points.append(("page", f"{did}:p{pno+1}:full",
                           {**base, "type": "page", "text": page_text}))

        # --- one meta point per doc: the full APA citation record ---
        meta_payload = {
            "doc_id": did, "title": title, "source_file": path.name,
            "total_pages": total_pages, "type": "meta",
            "doi": cite.get("doi"),
            "apa_inline": cite.get("inline"),
            "apa_narrative": cite.get("narrative"),
            "apa_reference": cite.get("reference"),
            "apa_authors": cite.get("authors_intext"),
            "year": cite.get("year"),
            "citation_unresolved": cite.get("unresolved", not cite),
            # embed the reference so the paper is findable by its bibliographic text
            "text": cite.get("reference") or f"{title} {cite.get('doi', '')}",
        }
        points.append(("meta", f"{did}:meta", meta_payload))
        doc.close()

        if not points:
            print(f"[warn] no extractable text in {path.name}, skipped")
            continue
        texts = [p[2]["text"] for p in points]
        vectors = list(embedder.embed(texts, batch_size=64))
        client.upsert(COLLECTION, points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                vector=vec.tolist(),
                payload=payload,
            )
            for (_, key, payload), vec in zip(points, vectors)
        ])
        print(f"[ok] {title[:55]!r}  doc_id={did}  pages={total_pages}  "
              f"chunks={n_chunks}  cite={cite.get('inline', 'UNRESOLVED')}")

    citations._save_cache(cite_cache)   # persist resolved DOIs for offline re-ingest
    ingest_notes(client, embedder)

    info = client.get_collection(COLLECTION)
    print(f"\nCollection '{COLLECTION}': {info.points_count} points total")


if __name__ == "__main__":
    main()
