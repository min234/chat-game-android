import json
import pathlib
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1) 경로 · 모델 설정
SRC        = "game.json"
OUT_DIR    = pathlib.Path("data2")
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "game_pairs.index"
META_PATH  = OUT_DIR / "meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 2) JSON 로드
records = json.load(open(SRC, encoding="utf-8"))

# 3) flat_records 생성
flat_records = []
for rec in records:
    section = rec["section"]
    for sub in rec.get("subsections", []):
        subsection = sub["title"]
        for text in sub.get("items", []):
            flat_records.append({
                "section": section,
                "subsection": subsection,
                "text": text
            })

# 4) DataFrame
df = pd.DataFrame(flat_records)
print("=== 샘플 레코드 ===")
print(df.head(), "\n")

# 5) 임베딩 생성
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")

# 6) FAISS 인덱스 구축
d = embeddings.shape[1]
index_flat = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap(index_flat)

ids = np.arange(len(df), dtype="int64")
index.add_with_ids(embeddings, ids)
print("벡터 개수:", index.ntotal)

# 7) 인덱스 저장
faiss.write_index(index, str(INDEX_PATH))
print("✅  인덱스 저장 →", INDEX_PATH)

# 8) meta.jsonl 쓰기
with META_PATH.open("w", encoding="utf-8") as f:
    for rec in flat_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("✅  메타 저장 →", META_PATH)
