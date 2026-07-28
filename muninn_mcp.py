#!/usr/bin/env python3
"""Muninn MCP server: page-citable retrieval over Ardian's research corpus.

Transports (env MUNINN_TRANSPORT):
    stdio (default)  — local use with Claude Desktop / Claude Code
    http             — Streamable HTTP for remote hosting (HF Spaces etc.)

Env vars:
    QDRANT_URL / QDRANT_API_KEY  — Qdrant Cloud; if unset, uses local ./qdrant_db
    MUNINN_TRANSPORT             — "stdio" | "http"
    MUNINN_PATH_TOKEN            — secret path segment for http mode
                                   (endpoint becomes /mcp-<token>)
    PORT                         — http port (HF Spaces sets 7860)
"""
import hashlib
import os
import re
from pathlib import Path

from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP, Image
from qdrant_client import QdrantClient, models

PDF_DIR = Path(os.environ.get("PDF_DIR", Path(__file__).parent / "pdfs"))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # must match ingest.py
COLLECTION = "muninn"

mcp = FastMCP(
    "muninn",
    instructions=(
        "Knowledge base of Ardian's research PDFs (machine unlearning, federated "
        "learning, LLM safety/poisoning, trustworthy AI). Every raw chunk carries "
        "an exact (source_file, page) citation - ALWAYS cite as 'filename, p. N'. "
        "type=claude_note results are Claude's prior reading notes (interpretation); "
        "type=raw results are verbatim paper text (ground truth). Prefer notes for "
        "orientation, raw chunks for quotes and verification. Each result also "
        "carries an APA-7 in-text citation (apa_citation); call get_citation(doc_id) "
        "for the full reference-list entry. The corpus is APA only — never mix "
        "citation styles.\n\n"
        "type=hyde results are GENERATED hypothetical documents (HyDE bridges), not "
        "corpus text — NEVER quote them as a source; they exist only to widen "
        "retrieval. RETRIEVAL-MISS PROTOCOL (top scores low / nothing relevant "
        "'indexed'): (1) retry search(note_type='everything') to check existing hyde "
        "bridges; (2) still weak → write a short hypothetical answer paragraph and "
        "call search(query, hypothesis=<that paragraph>) — HyDE-style: the hypothesis "
        "is embedded and averaged with the query, retrieving REAL raw/note chunks "
        "(Gao et al., 2023); (3) optionally save_hyde(query, hypothesis) to persist "
        "the bridge so the same miss becomes an indexed hit next time."
    ),
)

_client: QdrantClient | None = None
_embedder: TextEmbedding | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
        else:
            _client = QdrantClient(path=str(Path(__file__).parent / "qdrant_db"))
    return _client


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBED_MODEL)
    return _embedder


# note_type -> concrete payload types. "all" excludes hyde on purpose so
# ground-truth reads never mix in generated bridges; opt into them with
# "everything" or "hyde".
_TYPE_SETS = {
    "all": ["raw", "claude_note"],
    "everything": ["raw", "claude_note", "hyde"],
}


def _hyde_query_vector(query: str, hypothesis: str) -> list[float]:
    """HyDE query vector (Gao et al., 2023, Eq. 8, N=1): mean of the
    hypothetical-document embedding and the raw-query embedding.

    The hypothesis is document-like, so it's encoded with the passage encoder
    (embed); the query keeps its bge query-instruction prefix (query_embed).
    Both bge vectors share one cosine space, so the mean is well defined and
    hedges against a hallucinated hypothesis by keeping the literal query signal.
    """
    import numpy as np
    hvec = np.asarray(list(embedder().embed([hypothesis]))[0], dtype="float32")
    qvec = np.asarray(list(embedder().query_embed(query))[0], dtype="float32")
    return ((hvec + qvec) / 2.0).tolist()


@mcp.tool()
def search(query: str, top_k: int = 6, doc_id: str = "",
           note_type: str = "all", hypothesis: str = "") -> list[dict]:
    """Semantic search over the corpus. Returns chunks with page-level citations.

    Args:
        query: natural-language question or topic.
        top_k: number of results (default 6).
        doc_id: restrict to one document (from list_documents), "" = all.
        note_type: which payload types to search — "raw" (verbatim paper text,
            ground truth), "claude_note" (Claude's reading notes), "hyde"
            (generated HyDE bridges only), "all" (raw + claude_note, the
            default), or "everything" (raw + claude_note + hyde).
        hypothesis: HyDE fallback. When the plain query misses (low scores),
            pass a short hypothetical answer paragraph here. It is embedded and
            averaged with the query (Gao et al., 2023) to retrieve REAL corpus
            chunks via the hypothesis. Each hit is tagged retrieved_via="hyde".
            "" = ordinary query-vector search (retrieved_via="direct").
    """
    types = _TYPE_SETS.get(note_type, [note_type])
    must = [models.FieldCondition(key="type", match=models.MatchAny(any=types))]
    if doc_id:
        must.append(models.FieldCondition(key="doc_id",
                                          match=models.MatchValue(value=doc_id)))
    if hypothesis.strip():
        qvec = _hyde_query_vector(query, hypothesis)
        via = "hyde"
    else:
        qvec = list(embedder().query_embed(query))[0].tolist()
        via = "direct"
    hits = client().query_points(
        COLLECTION, query=qvec, limit=top_k,
        query_filter=models.Filter(must=must), with_payload=True,
    ).points
    out = []
    for h in hits:
        p = h.payload
        if p["type"] == "hyde":
            # a generated bridge, not corpus text — must never be cited
            cite = "HyDE bridge (generated hypothesis — NOT a source; verify against raw)"
            apa = None
        elif p.get("page"):
            cite = f"{p['source_file']}, p. {p['page']}"
            # APA in-text: add the page locator for a verbatim raw hit —
            # "(Author, Year)" -> "(Author, Year, p. N)".
            apa = p.get("apa_inline")
            if apa and apa.endswith(")"):
                apa = f"{apa[:-1]}, p. {p['page']})"
        else:
            cite = f"{p['source_file']} (Claude note)"
            apa = p.get("apa_inline")
        out.append({
            "score": round(h.score, 4),
            "type": p["type"],
            "retrieved_via": via,      # "direct" query vector or "hyde" fallback
            "doc_id": p["doc_id"],
            "title": p["title"],
            "citation": cite,          # file+page locator (ground truth)
            "apa_citation": apa,       # APA-7 in-text citation (None for hyde)
            "doi": p.get("doi"),
            "page": p.get("page"),
            "text": p["text"],
        })
    return out


@mcp.tool()
def save_hyde(query: str, hypothesis: str, doc_id: str = "",
              title: str = "") -> dict:
    """Persist a HyDE hypothetical document as a reusable type=hyde bridge.

    Use after a search(query, hypothesis=...) fallback produced good grounding,
    so the same retrieval miss becomes an indexed hit next time (searchable via
    note_type="hyde" or "everything"). A hyde bridge is generated text, never a
    citable source. Idempotent per (doc_id, query) — re-saving updates in place.

    Args:
        query: the user query this hypothesis answers (used for the bridge id).
        hypothesis: the hypothetical answer paragraph to store and embed.
        doc_id: optional document this bridge is about ("" = corpus-wide).
        title: optional label ("" = derived from the query).
    """
    import time
    import uuid
    vec = list(embedder().embed([hypothesis]))[0].tolist()
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hyde:{doc_id}:{query}"))
    payload = {
        "doc_id": doc_id or "hyde-bridge",
        "title": title or f"HyDE bridge: {query[:60]}",
        "source_file": "(hyde hypothesis — not a source)",
        "type": "hyde",
        "query": query,
        "text": hypothesis,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    client().upsert(COLLECTION, points=[models.PointStruct(
        id=pid, vector=vec, payload=payload)])
    return {"saved_id": pid, "type": "hyde", "doc_id": payload["doc_id"],
            "query": query,
            "note": "retrievable via search(note_type='hyde'|'everything'); "
                    "never quote as a source"}


@mcp.tool()
def list_documents() -> list[dict]:
    """List all documents in the knowledge base with doc_id, title, and pages."""
    docs: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client().scroll(
            COLLECTION, limit=500, offset=offset, with_payload=True,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="type",
                match=models.MatchAny(any=["page", "claude_note", "meta"]))]),
        )
        for pt in points:
            p = pt.payload
            d = docs.setdefault(p["doc_id"], {
                "doc_id": p["doc_id"], "title": p["title"],
                "source_file": p["source_file"], "total_pages":
                p.get("total_pages"), "has_claude_note": False,
                "doi": None, "apa_inline": None, "year": None})
            if p["type"] == "claude_note":
                d["has_claude_note"] = True
            if p["type"] == "meta":
                d["doi"] = p.get("doi")
                d["apa_inline"] = p.get("apa_inline")
                d["year"] = p.get("year")
            if p.get("total_pages"):
                d["total_pages"] = p["total_pages"]
        if offset is None:
            break
    return sorted(docs.values(), key=lambda d: d["title"])


@mcp.tool()
def get_page(doc_id: str, page: int) -> dict:
    """Full text of one page of a document — for reading context around a search hit.

    Args:
        doc_id: document id from search/list_documents.
        page: 1-based page number.
    """
    points, _ = client().scroll(
        COLLECTION, limit=1, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="type", match=models.MatchValue(value="page")),
            models.FieldCondition(key="page", match=models.MatchValue(value=page)),
        ]),
    )
    if not points:
        return {"error": f"no page {page} for doc {doc_id}"}
    p = points[0].payload
    return {"title": p["title"], "citation": f"{p['source_file']}, p. {page}",
            "page": page, "total_pages": p.get("total_pages"), "text": p["text"]}


@mcp.tool()
def get_citation(doc_id: str) -> dict:
    """APA-7 citation for a document: in-text (parenthetical + narrative) and
    the full reference-list entry. Use these verbatim when writing — the corpus
    is APA only, never mix styles. For a direct quote, add the page to the
    in-text form: '(Author, Year, p. N)'.

    Args:
        doc_id: document id from search/list_documents.
    """
    points, _ = client().scroll(
        COLLECTION, limit=1, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="type", match=models.MatchValue(value="meta")),
        ]),
    )
    if not points:
        return {"error": f"no citation metadata for doc {doc_id} "
                         f"(re-run ingest.py to build it)"}
    p = points[0].payload
    return {
        "doc_id": doc_id,
        "title": p["title"],
        "source_file": p["source_file"],
        "doi": p.get("doi"),
        "apa_in_text": p.get("apa_inline"),
        "apa_narrative": p.get("apa_narrative"),
        "apa_reference": p.get("apa_reference"),
        "unresolved": p.get("citation_unresolved", False),
    }


def _pdf_for(doc_id: str) -> Path | None:
    for p in PDF_DIR.glob("*.pdf"):
        if hashlib.sha1(p.name.encode()).hexdigest()[:12] == doc_id:
            return p
    return None


_LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
        "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"',
        "”": '"', "­": ""}


def _norm_token(w: str) -> str:
    for k, v in _LIG.items():
        w = w.replace(k, v)
    return re.sub(r"[^0-9a-z]+", "", w.lower())


def _fuzzy_rects(pg, needle: str) -> list:
    """Locate needle by word-sequence alignment — tolerates hyphenation,
    ligatures, punctuation, and small edits. Needs >=50% of words to align."""
    import difflib

    import fitz  # PyMuPDF

    words = pg.get_text("words")  # (x0, y0, x1, y1, word, ...)
    page_toks = [_norm_token(w[4]) or f"\x00{i}" for i, w in enumerate(words)]
    need_toks = [t for t in (_norm_token(t) for t in needle.split()) if t]
    if len(need_toks) < 3:
        return []
    sm = difflib.SequenceMatcher(None, page_toks, need_toks, autojunk=False)
    idxs, matched = [], 0
    for a, _, size in sm.get_matching_blocks():
        if size >= 3:  # ignore incidental 1-2 word matches
            matched += size
            idxs.extend(range(a, a + size))
    if matched < 0.5 * len(need_toks):
        return []
    return [fitz.Rect(words[i][:4]) for i in idxs]


def _locate(pg, text: str) -> tuple[str, list]:
    """Find text on a page: exact -> sentence pieces -> fuzzy word alignment."""
    rects = pg.search_for(text)
    if rects:
        return "exact", rects
    pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text)
              if len(s.strip()) > 15]
    rects = [r for p in pieces for r in pg.search_for(p)]
    if rects:
        return "sentences", rects
    rects = _fuzzy_rects(pg, text)
    return ("fuzzy" if rects else "none"), rects


def _best_chunk_on_page(doc_id: str, page: int, query: str,
                        min_score: float = 0.5) -> str | None:
    """Semantically nearest raw chunk on one page (for paraphrased highlights)."""
    qvec = list(embedder().query_embed(query))[0].tolist()
    hits = client().query_points(
        COLLECTION, query=qvec, limit=1, with_payload=True,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="type", match=models.MatchValue(value="raw")),
            models.FieldCondition(key="page", match=models.MatchValue(value=page)),
        ]),
    ).points
    if hits and hits[0].score >= min_score:
        return hits[0].payload["text"]
    return None


@mcp.tool()
def render_page(doc_id: str, page: int, highlight: str = ""):
    """Render a PDF page as an image, optionally with a passage highlighted.
    Use after search/get_page to SHOW the user the exact source location.

    Args:
        doc_id: document id from search/list_documents.
        page: 1-based page number.
        highlight: passage to highlight. Verbatim raw-chunk text from search()
            always works; paraphrases usually work too — matching falls back
            exact -> sentence pieces -> fuzzy word alignment -> semantic
            (embeds the text and highlights the nearest raw chunk on this
            page). Returns the image plus a highlight_method report.
            "" = no highlighting.
    """
    import fitz  # PyMuPDF

    path = _pdf_for(doc_id)
    if path is None:
        return {"error": f"no PDF on this server for doc_id {doc_id} "
                         f"(PDF_DIR={PDF_DIR})"}
    doc = fitz.open(path)
    if not 1 <= page <= doc.page_count:
        return {"error": f"page {page} out of range 1..{doc.page_count}"}
    pg = doc[page - 1]

    method, rects = "none", []
    if highlight:
        method, rects = _locate(pg, highlight)
        if not rects:  # last resort: semantic match against this page's chunks
            chunk = _best_chunk_on_page(doc_id, page, highlight)
            if chunk:
                _, rects = _locate(pg, chunk)
                if rects:
                    method = "semantic"
        for r in rects:
            pg.add_highlight_annot(r)

    pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom ≈ 1200x1600
    png = pix.tobytes("png")
    doc.close()
    img = Image(data=png, format="png")
    if not highlight:
        return img
    info = {"highlight_method": method, "n_rects": len(rects)}
    if method == "none":
        info["hint"] = ("no match found on this page — retry with the "
                        "verbatim 'text' of a raw search() hit for this page")
    return [img, info]


def main() -> None:
    transport = os.environ.get("MUNINN_TRANSPORT", "stdio")
    if transport == "http":
        token = os.environ.get("MUNINN_PATH_TOKEN", "")
        if not token:
            raise SystemExit("Set MUNINN_PATH_TOKEN for http mode (capability URL).")
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT", "7860"))
        mcp.settings.streamable_http_path = f"/mcp-{token}"
        mcp.settings.stateless_http = True   # simpler for serverless-ish hosts
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
