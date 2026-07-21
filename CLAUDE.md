# Muninn — Research Knowledge Base (read this first)

Page-citable RAG knowledge base over Ardian's research PDFs
The PDFs are **already read, chunked, embedded, and indexed**. Do NOT re-read
PDFs to answer content questions — use the **muninn MCP server**.

## Architecture

```
pdfs/ (papers, DOI-prefixed filenames; doc_id = sha1(filename)[:12])
  → ingest.py    PyMuPDF page-level extraction → ~1000-char chunks that never
                 cross page boundaries → bge-small-en-v1.5 (fastembed, 384-dim)
  → qdrant_db/   embedded Qdrant, collection "muninn" (Cloud later via
                 push_to_cloud.py; env QDRANT_URL/QDRANT_API_KEY switches)
  → muninn_mcp.py FastMCP server; stdio locally, Streamable HTTP + secret-path
                 token on HF Spaces (see Dockerfile + README)
```

Three payload types per document: `raw` = verbatim chunk with exact
(source_file, page) — ground truth; `page` = full page text; `claude_note` =
Claude's stored reading notes (problem/method/findings/limitations/
cross-links, page-cited quotes) from `notes/*.jsonl`, incl. a corpus-wide
synthesis (search "corpus synthesis").

## MCP tools

- `search(query, top_k, doc_id, note_type)` — semantic search with citations
- `list_documents()` — corpus inventory
- `get_page(doc_id, page)` — full page text around a hit
- `render_page(doc_id, page, highlight)` — actual PDF page as image with the
  passage highlighted; use it to show the source

## Rules

- Always cite as `filename, p. N`. Notes are interpretation; verify important
  claims against `raw` chunks.
- Retrieval miss? Retry HyDE-style: write a hypothetical answer paragraph and
  search with that (the technique is itself in the corpus, ACL 2023).
- New PDF added to `pdfs/`? Run `uv run ingest.py` (idempotent per file),
  write a reading note into `notes/` (JSONL, same schema as existing), re-run
  ingest, and `push_to_cloud.py` if hosted.

## Dev notes

- `uv sync` installs deps (pyproject.toml, Python >=3.10,<3.14 — onnxruntime
  has no reliable 3.14 wheels).
- Local Qdrant is single-process: stop the MCP server before running
  ingest.py.
- Renaming a PDF changes its doc_id — update notes/*.jsonl references and
  rebuild the DB (delete qdrant_db/, re-ingest).
- EMBED_MODEL must stay identical in ingest.py and muninn_mcp.py; changing it
  requires full re-ingest (vectors from different models are incompatible).
- Never hand-edit qdrant_db/ — it's a build artifact.
