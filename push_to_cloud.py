#!/usr/bin/env python3
"""Migrate the local Qdrant DB to Qdrant Cloud. Run once, from your machine.

Usage:
    QDRANT_URL=https://xxxx.cloud.qdrant.io QDRANT_API_KEY=... uv run push_to_cloud.py
"""
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

COLLECTION = "muninn"
BATCH = 200


def main() -> None:
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url or not key:
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY (from Qdrant Cloud console).")

    local = QdrantClient(path=str(Path(__file__).parent / "qdrant_db"))
    cloud = QdrantClient(url=url, api_key=key)

    info = local.get_collection(COLLECTION)
    if not cloud.collection_exists(COLLECTION):
        cloud.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=info.config.params.vectors.size,
                distance=info.config.params.vectors.distance,
            ),
        )
        for field, schema in [("doc_id", models.PayloadSchemaType.KEYWORD),
                              ("type", models.PayloadSchemaType.KEYWORD),
                              ("page", models.PayloadSchemaType.INTEGER)]:
            cloud.create_payload_index(COLLECTION, field, schema)

    moved = 0
    offset = None
    while True:
        points, offset = local.scroll(COLLECTION, limit=BATCH, offset=offset,
                                      with_payload=True, with_vectors=True)
        if not points:
            break
        cloud.upsert(COLLECTION, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ])
        moved += len(points)
        print(f"pushed {moved}/{info.points_count}")
        if offset is None:
            break
    print("done. Verify:", cloud.get_collection(COLLECTION).points_count, "points in cloud")


if __name__ == "__main__":
    main()
