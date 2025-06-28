# src/core/semantic/code_search.py
"""
Light-weight FAISS wrapper for “mini-CodeSearchNet”.
---------------------------------------------------
• vectors :  data/embeddings/csn_mini.index      (built by embed_csn_faiss.py)
• metadata:  data/embeddings/csn_mini.meta.jsonl (same order as index)
"""

from __future__ import annotations
import json, pathlib
import numpy as np
from config import settings
import faiss                             # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

INDEX_F = settings.EMBEDDINGS_DIR / "csn_mini.index"
META_F  = settings.EMBEDDINGS_DIR / "csn_mini.meta.jsonl"

# ---------- load once on first import ----------
_model  = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL, device="cpu")
_index  = faiss.read_index(str(INDEX_F))

_meta: list[dict] = []
with open(META_F) as fp:
    for line in fp:
        _meta.append(json.loads(line))

# ---------- public api ----------
def search(code_snippet: str, k: int = 5) -> list[tuple[float, dict]]:
    """
    Return top-k triple (similarity, metadata) where similarity ∈ [0,1].
    """
    if not code_snippet.strip():
        return []

    vec = _model.encode(
        [code_snippet],
        normalize_embeddings=True
    ).astype(np.float32)
    # FAISS similarity = 1 - cosine_distance/2  (because vectors are L2-normed)
    D, I = _index.search(vec, k)
    return [(float(1 - d / 2), _meta[i]) for d, i in zip(D[0], I[0])]
