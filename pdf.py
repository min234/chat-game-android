import os
from langchain.document_loaders import CSVLoader, PyPDFLoader, BSHTMLLoader
from dotenv import load_dotenv
import json
import pathlib
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

pdf_loader = PyPDFLoader(file_path='EventGameDocument.pdf')

# 문서 로드
pdf_documents = pdf_loader.load()
cleaned_pdf_content = pdf_documents[0].page_content.replace('\n', ' ')

# 로드된 문서 확인
print("\nPDF 문서 내용:")
print("\n정제된 PDF 문서 내용:")
print(cleaned_pdf_content)


# ---------- 1) 경로 · 모델 설정 ----------
SRC         = "styled_questions.json"
OUT_DIR     = pathlib.Path("data")
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH  = OUT_DIR / "sent_pairs.index"
META_PATH   = OUT_DIR / "meta.jsonl"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
 
# ---------- 5) 임베딩 추출 ----------
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    df["text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype("float32")

# ---------- 6) FAISS 인덱스 생성 ----------
d = embeddings.shape[1]
index_flat = faiss.IndexFlatIP(d)
index = faiss.IndexIDMap(index_flat)

ids = df["id"].values.astype("int64")
index.add_with_ids(embeddings, ids)
print("   벡터 개수:", index.ntotal)

# ---------- 7) 인덱스 저장 ----------
faiss.write_index(index, str(INDEX_PATH))
print("✅  인덱스 저장 →", INDEX_PATH)

# ---------- 8) meta.jsonl 새로 쓰기 ----------
with META_PATH.open("w", encoding="utf-8") as f:
    for rec in new_rows:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("✅  메타 저장 →", META_PATH)
