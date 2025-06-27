# src/data_ingestion/codesearchnet_ingest.py
"""
Download *one* language slice of CodeSearchNet and extract only the
<func-docstring, repo, path> triples we need for contrastive fine-tuning.

Default: python + Go  → ≈ 14 GB compressed → ~6 GB after filtering.
"""
import gzip, json, pathlib, subprocess, random, shutil, tarfile, textwrap
from typing import Iterator, Dict

import sys, pathlib

from datasets import load_dataset, disable_caching # type: ignore
disable_caching()   

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings



# S3 base (official) :contentReference[oaicite:0]{index=0}
S3_ROOT = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2"
LANGS = ["python", "go", "javascript"]           # keep it light; add more later

import sys
if len(sys.argv) > 1:
    LANGS = (
        [l.strip() for l in sys.argv[1].split(",")]  # “python,java”
        if len(sys.argv) == 2 else sys.argv[1:]      # “python java”
    )

RAW_DIR = settings.DATA_DIR / "raw/csn"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def stream_jsonl_gz(path: pathlib.Path) -> Iterator[Dict]:
    with gzip.open(path, "rt") as gz:
        for line in gz:
            try:
                yield json.loads(line)
            except Exception:
                continue

def download_language(lang: str) -> None:
    """
    Stream CodeSearchNet `train` split for <lang> from the HuggingFace hub,
    keep ~1 % of rows, and write a newline-JSON file with *only* the fields
    we need for contrastive fine-tuning.

      output:  data/raw/csn/<lang>.mini.jsonl               (≈ 250 MB / lang)
    """
    out = RAW_DIR / f"{lang}.mini.jsonl"
    if out.exists():
        print(f"▶ {lang}: subset already exists → {out}")
        return

    print(f"▶ {lang}: streaming train split from 🤗 hub…")
    ds = load_dataset("code_search_net", lang, split="train", streaming=True)

    kept, total = 0, 0
    with open(out, "w") as fp:
        for rec in ds:
            total += 1
            if random.random() < 0.01:            # keep 1 %
                # ── best-effort repo / path recovery (optional) ────────────
                repo = rec.get("repo") or rec.get("repo_name")
                path = rec.get("path")
                if (not repo or not path) and rec.get("url"):
                    parts = rec["url"].split("/")
                    if len(parts) > 6:
                        repo = repo or f"{parts[3]}/{parts[4]}"
                        path = path or "/".join(parts[7:]).split("#")[0]

                fp.write(json.dumps({
                    "func_code": (
                        rec.get("code")
                        or rec.get("func_code")
                        or rec.get("func_code_string")
                    ),
                    "func_doc":  (
                        rec.get("docstring")
                        or rec.get("func_docstring")
                        or rec.get("func_documentation_string")
                    ),
                    "repo": repo or "unknown",
                    "path": path or "unknown",
                }) + "\n")
                kept += 1

            if total % 100_000 == 0:
                print(f"  …{total:,} streamed, {kept:,} kept")

    print(f"✅ {lang}: wrote {kept:,} / {total:,} rows → {out}")


if __name__ == "__main__":
    for lang in LANGS:
        download_language(lang.strip())

    # free disk
    for tar in RAW_DIR.glob("*.tar.gz"):
        print(f"🗑  deleting {tar.name} to reclaim {tar.stat().st_size/1e9:.1f} GB")
        tar.unlink()

    print("✅ CodeSearchNet mini-subset ready.")
