import json, faiss, numpy as np, asyncio
from pathlib import Path
from sentence_transformers import SentenceTransformer
 
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

INDEX = faiss.read_index("data/sent_pairs.index")
meta  = {rec["id"]: rec for rec in map(json.loads, open("data/meta.jsonl", encoding="utf-8"))}
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ────────────────────────────────
# 1. 벡터 검색
def search(query: str, k: int = 1):
    vec = embed.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = INDEX.search(vec, k)
    return [
        {
            "score": float(scores[0][i]),
            **meta[int(idx)]      # id/text/Black/White
        } for i, idx in enumerate(ids[0]) if idx != -1
    ]
    