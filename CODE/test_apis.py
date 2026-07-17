"""
Quick smoke-test for Azure embedding + Cohere rerank APIs.
Run: python test_apis.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings

settings = Settings.from_env()

# ── 1. Azure Embedding ────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Azure Embedding API")
print(f"  endpoint : {settings.embed_model_endpoint}/openai/v1/embeddings")
print(f"  model    : {settings.embed_model}")
print(f"  api-key  : {settings.embed_model_api_key[:8]}…")

try:
    from src.embeddings_index import embed_texts
    vecs = embed_texts(["Data analyst with Python and SQL experience"], settings)
    dim = len(vecs[0])
    print(f"  PASS  — got 1 vector, dim={dim}")
except Exception as exc:
    print(f"  FAIL  — {exc}")
    sys.exit(1)

# ── 2. Cohere Rerank ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: Cohere Rerank API (via Azure APIM)")
print(f"  endpoint : {settings.cohere_rerank_endpoint}")
print(f"  model    : {settings.cohere_rerank_model}")
print(f"  api-key  : {settings.cohere_rerank_api_key[:8] if settings.cohere_rerank_api_key else 'NOT SET'}…")

import json, urllib.request, urllib.error

payload = {
    "model": settings.cohere_rerank_model,
    "query": "data analyst",
    "documents": [
        "Data Scientist — builds ML models and statistical analyses",
        "Software Engineer — writes backend code and APIs",
        "Business Intelligence Analyst — creates dashboards and reports",
    ],
    "top_n": 3,
}
headers = {
    "Content-Type": "application/json",
    "api-key": settings.cohere_rerank_api_key,
}

body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(settings.cohere_rerank_endpoint, data=body, headers=headers, method="POST")
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    rankings = result.get("results", [])
    print(f"  PASS  — got {len(rankings)} rankings")
    for r in rankings:
        print(f"          index={r.get('index')} score={r.get('relevance_score', '?'):.4f}")
except urllib.error.HTTPError as exc:
    body_text = exc.read().decode("utf-8", errors="replace")
    print(f"  FAIL  — HTTP {exc.code}: {body_text[:300]}")
    sys.exit(1)
except Exception as exc:
    print(f"  FAIL  — {exc}")
    sys.exit(1)

print()
print("=" * 60)
print("All API checks passed. Safe to run: python build_graph.py --embed")
