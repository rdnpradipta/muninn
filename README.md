# Muninn — Page-Citable Research Knowledge Base + MCP Server

Named after Odin's raven of memory. From research PDFs → chunked at page level →
embedded (`BAAI/bge-small-en-v1.5`) → Qdrant vector DB → served over MCP so any
future Claude session (desktop, web, phone) can query the corpus **with exact
`filename, p. N` citations** — no re-reading, no chat-context cost.

> **Note:** `pdfs/`, `notes/`, and `qdrant_db/` are not included in this repo.
> `pdfs/` holds the source papers, `notes/` holds Claude's generated reading
> notes, and `qdrant_db/` is the resulting vector DB — all excluded for
> privacy, since they contain personal research material and its derived
> content rather than shareable code.


## Layout

```
pdfs/            the source PDFs
notes/           Claude's reading notes (notes*.jsonl) — re-ingestable
ingest.py        PDFs + notes → qdrant_db/  (run once, and after adding PDFs)
muninn_mcp.py    MCP server (stdio locally, Streamable HTTP when hosted)
push_to_cloud.py local qdrant_db/ → Qdrant Cloud (one command)
Dockerfile       Hugging Face Space deployment
qdrant_db/       the vector DB (created by ingest.py)
```

Data model (collection `muninn`): `type=raw` — verbatim chunks, ~1000 chars,
never crossing page boundaries → every chunk has exactly one `(source_file,
page)`; `type=page` — full page text (backs `get_page`, and later
page-rendering/highlighting: exact `search_for` → sentence pieces → fuzzy
word alignment → semantic nearest-chunk fallback); `type=claude_note` —
interpretation layer. Raw = ground truth, notes = orientation.

## Step 1 — Ingest (your machine, ~2-3 min)

```bash
cd <this folder>
uv run ingest.py          # downloads bge-small (~130MB) once, builds qdrant_db/
```

## Step 2 — Use locally right away (optional)

Claude Desktop → `claude_desktop_config.json`:

```json
{"mcpServers": {"muninn": {"command": "uv",
  "args": ["run", "--directory", "/ABSOLUTE/PATH/TO/THIS/FOLDER", "muninn_mcp.py"]}}}
```

Claude Code: `claude mcp add muninn -- uv run --directory /ABSOLUTE/PATH muninn_mcp.py`

Note: Qdrant local mode is single-process — stop the MCP server before
re-running `ingest.py`.

## Step 3 — Host it (laptop-off access, $0/mo)

**3a. Qdrant Cloud (free tier, 4GB):** create a cluster at cloud.qdrant.io,
copy URL + API key, then:

```bash
QDRANT_URL=https://xxxx.cloud.qdrant.io QDRANT_API_KEY=... uv run push_to_cloud.py
```

**3b. HF Space:** create a Space (SDK = **Docker**), push `Dockerfile` +
`muninn_mcp.py` to it. In Space **Settings → Variables and secrets** set:

| name | kind | value |
|---|---|---|
| `MUNINN_PATH_TOKEN` | secret | `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `QDRANT_URL` | variable | your cluster URL |
| `QDRANT_API_KEY` | secret | your cluster key |

**3c. Connect Claude:** Settings → Connectors → Add custom connector →
`https://<user>-<space>.hf.space/mcp-<TOKEN>`

## MCP tools

`search(query, top_k, doc_id, note_type)` → ranked chunks with `citation`;
`list_documents()` → corpus inventory; `get_page(doc_id, page)` → full page
text. Retrieval tip: if a query misses, try HyDE (in the corpus, doc
`322d71209a2f`): write a hypothetical paragraph answering the question and
search with that.

## Adding papers later

Drop PDFs into `pdfs/`, re-run `uv run ingest.py` (idempotent per file), then
`push_to_cloud.py` if hosted. Ask Claude to read the new paper and append a
note to `notes/` for the interpretation layer.
