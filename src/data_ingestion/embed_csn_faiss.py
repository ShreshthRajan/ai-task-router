# src/data_ingestion/embed_csn_faiss.py
"""
Embed CodeSearchNet mini-subset (all languages) and build a FAISS index
for fast semantic search.  Output:
  data/embeddings/csn_mini.index
  data/embeddings/csn_mini.meta.jsonl
"""

import json, pathlib, sys, time, math
from typing import List

from matplotlib.pylab import f
import faiss                      # pip install faiss-cpu
from tqdm import tqdm             # pip install tqdm
from sentence_transformers import SentenceTransformer
from src.config import settings   # works because we run `python -m …`

RAW_DIR  = settings.DATA_DIR / "raw"  / "csn"
OUT_DIR  = settings.EMBEDDINGS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = OUT_DIR / "csn_mini.index"
META_FILE  = OUT_DIR / "csn_mini.meta.jsonl"

model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL,
                            device="cpu")      # GPU optional

def batched(iterable, n=64):
    chunk: List[str] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def stream_all_code():
    """
    Yield records **with the language already attached** so the metadata
    writer doesn't need the outer file handle.
    """
    for f in RAW_DIR.glob("*.mini.jsonl"):
        lang = f.stem.split(".")[0]            #  python / go / javascript …
        with open(f) as fp:
            for line in fp:
                rec = json.loads(line)
                rec["lang"] = lang             # ← add here
                yield rec

def main():
    print("▶ building embeddings…")
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)

    with open(META_FILE, "w") as meta_fp:
        total = sum(1 for _ in stream_all_code())   # only to show progress
        done  = 0
        for chunk in batched(stream_all_code(), 64):
            codes = [rec["func_code"] for rec in chunk]
            embs  = model.encode(codes, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
            index.add(embs)

            # save minimal metadata (1 line per vector, same order)
            for rec in chunk:
                meta_fp.write(json.dumps({
                    "repo": rec.get("repo")  or "unknown",
                    "path": rec.get("path")  or "unknown",
                    "lang": rec.get("lang")  or "unknown",
                }) + "\n")

            done += len(chunk)
            if done % 1000 == 0 or done == total:
                pct = done / total * 100
                print(f"  …{done:,}/{total:,}  ({pct:4.1f} %)", end="\r")

    faiss.write_index(index, str(INDEX_FILE))
    print(f"\n✅ wrote {index.ntotal:,} vectors → {INDEX_FILE}")
    print(f"✅ metadata lines match: {sum(1 for _ in open(META_FILE)):,}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"⏱  finished in {time.time()-t0:,.1f}s")
