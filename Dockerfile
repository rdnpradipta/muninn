# Muninn MCP server — Hugging Face Space (Docker SDK)
# Space settings: SDK = Docker. Port 7860 is the HF default.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastembed qdrant-client "mcp>=1.9" pymupdf

# Pre-download the embedding model into the image so cold starts are fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY muninn_mcp.py .
# PDFs shipped with the server so render_page works remotely (~45MB)
COPY pdfs/ ./pdfs/

ENV MUNINN_TRANSPORT=http
# Set in HF Space settings (Settings -> Variables and secrets):
#   MUNINN_PATH_TOKEN  (secret)  e.g. output of: python -c "import secrets; print(secrets.token_urlsafe(24))"
#   QDRANT_URL         (variable)
#   QDRANT_API_KEY     (secret)

EXPOSE 7860
CMD ["python", "muninn_mcp.py"]
