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

Five payload types: `raw` = verbatim chunk with exact (source_file, page) —
ground truth; `page` = full page text; `meta` = one per document, the APA-7
citation record; `claude_note` = Claude's stored reading notes
(problem/method/findings/limitations/cross-links, page-cited quotes) from
`notes/*.jsonl`, incl. a corpus-wide synthesis (search "corpus synthesis");
`hyde` = a **generated** HyDE hypothetical document (a retrieval bridge saved
by `save_hyde`), NOT corpus text and NEVER a citable source — it only widens
recall. `hyde` is excluded from the default `search` (note_type="all"); opt in
with note_type="hyde" or "everything".

## Citations (APA 7 only — never mix styles)

Every PDF is named `<url-encoded-DOI><sep><title>.pdf` (`%2F`→`/`, `<sep>` is
`_` or ` - `). `citations.py` extracts the DOI and resolves authoritative
metadata via **DOI.org content negotiation** (CSL-JSON), which routes each DOI
to its owning agency — **Crossref** (Springer/IEEE/ACL/ACM) or **DataCite**
(arXiv `10.48550/*`) — so one path covers the whole corpus. Resolutions are
cached in `.citation_cache.json` (gitignored) so re-ingest is offline.
`ingest.py` stores `doi` + `apa_inline` on every raw/page point and a full
`meta` point (inline, narrative, reference, authors, year). Offline / 404 →
recorded as `citation_unresolved`.

## MCP tools

- `search(query, top_k, doc_id, note_type, hypothesis)` — semantic search with
  citations. `note_type`: raw | claude_note | hyde | all (raw+note, default) |
  everything (+hyde). `hypothesis`: HyDE fallback — pass a hypothetical answer
  paragraph to retrieve real chunks via the averaged hypothesis+query vector;
  hits are tagged `retrieved_via="hyde"`.
- `save_hyde(query, hypothesis, doc_id, title)` — persist a good hypothesis as a
  reusable `hyde` bridge (idempotent per doc_id+query) so the same miss becomes
  an indexed hit next time
- `list_documents()` — corpus inventory
- `get_page(doc_id, page)` — full page text around a hit
- `get_citation(doc_id)` — APA-7 in-text (parenthetical + narrative) and full
  reference-list entry for a document
- `render_page(doc_id, page, highlight)` — actual PDF page as image with the
  passage highlighted; use it to show the source

## Rules

- Always cite as `filename, p. N` for source location. For writing, use the
  APA-7 forms: `apa_citation` on each search hit, `get_citation()` for the
  reference list. Quote → add the page: `(Author, Year, p. N)`. APA only.
- Notes are interpretation; verify important claims against `raw` chunks.
- Retrieval miss (low scores / nothing relevant "indexed")? Follow the fallback
  protocol: (1) retry `search(note_type="everything")` to check existing `hyde`
  bridges; (2) still weak → write a hypothetical answer paragraph and call
  `search(query, hypothesis=<paragraph>)` — HyDE-style, retrieves real chunks
  via the averaged hypothesis+query vector (technique itself is in the corpus,
  ACL 2023); (3) if it grounded well, `save_hyde(query, hypothesis)` to persist
  the bridge. Never quote a `hyde` result as a source — verify against `raw`.
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
- The citation schema (doi/apa/meta points) needs a full rebuild: delete
  qdrant_db/ and re-run ingest.py. First run needs network (to resolve DOIs);
  afterwards `.citation_cache.json` makes it offline. A DOI that won't resolve
  is stored as `citation_unresolved` — fix the filename's DOI or the network
  and re-ingest that file.
