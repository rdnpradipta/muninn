#!/usr/bin/env python3
"""Dump the per-document citation records (type=meta) from the DB.

    uv run inspect_citations.py            # table: file | year | in-text | doi
    uv run inspect_citations.py --refs     # also print the full APA reference
    uv run inspect_citations.py --bad      # only unresolved DOIs

Reads the same DB as the server (local ./qdrant_db, or QDRANT_URL if set).
NOTE: local Qdrant is single-process — stop the MCP server before running.
"""
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

COLLECTION = "muninn"


def client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(path=str(Path(__file__).parent / "qdrant_db"))


def main() -> None:
    show_refs = "--refs" in sys.argv
    only_bad = "--bad" in sys.argv

    rows, offset = [], None
    c = client()
    while True:
        points, offset = c.scroll(
            COLLECTION, limit=500, offset=offset, with_payload=True,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="type", match=models.MatchValue(value="meta"))]),
        )
        rows.extend(p.payload for p in points)
        if offset is None:
            break

    rows.sort(key=lambda p: (p.get("year") or "", p.get("source_file") or ""))
    shown = 0
    for p in rows:
        if only_bad and not p.get("citation_unresolved"):
            continue
        shown += 1
        flag = "  !! UNRESOLVED" if p.get("citation_unresolved") else ""
        year = p.get("year") or "----"
        intext = p.get("apa_inline") or "(none)"
        print(f"{year}  {intext:28}  doi={p.get('doi')}{flag}")
        print(f"        file: {p.get('source_file')}")
        if show_refs:
            print(f"        ref : {p.get('apa_reference') or '(unresolved)'}")
    print(f"\n{shown} document(s)"
          + (" with unresolved citations" if only_bad else " total"))


if __name__ == "__main__":
    main()
