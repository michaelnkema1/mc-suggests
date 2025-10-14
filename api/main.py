from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

app = FastAPI(title="MC-Suggests API")

# Lazy globals
DF = None
VECTORIZER = None
TFIDF_X = None
SBERT_EMB = None


def load_df():
    global DF
    if DF is None:
        DF = pd.read_parquet("mangadex_clean.parquet")
    return DF


def load_tfidf():
    global VECTORIZER, TFIDF_X
    if VECTORIZER is None or TFIDF_X is None:
        VECTORIZER = joblib.load("models/tfidf_vectorizer.joblib")
        TFIDF_X = sparse.load_npz("models/tfidf_matrix.npz")
    return VECTORIZER, TFIDF_X


def load_sbert():
    global SBERT_EMB
    if SBERT_EMB is None:
        SBERT_EMB = np.load("models_sbert/embeddings.npy")
    return SBERT_EMB


def find_by_title(df: pd.DataFrame, query: str, k: int = 5) -> List[int]:
    q = query.strip().lower()
    mask = df["title_lc"].fillna("").str.contains(re.escape(q))
    idxs = np.where(mask.values)[0].tolist()
    if not idxs and q:
        exact = np.where(df["title_lc"].fillna("").values == q)[0].tolist()
        idxs = exact
    return idxs[:k]


class RecommendResponseItem(BaseModel):
    id: str
    title: Optional[str] = ""
    score: float
    year: Optional[int] = None
    rating: Optional[float] = None


class RecommendResponse(BaseModel):
    seed_count: int
    results: List[RecommendResponseItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/tfidf", response_model=RecommendResponse)
def recommend_tfidf(query: str = Query(...), k: int = Query(10, ge=1, le=50)):
    df = load_df()
    _, X = load_tfidf()
    seed_idxs = find_by_title(df, query, k=5)
    if not seed_idxs:
        return {"seed_count": 0, "results": []}
    seed_mat = X[seed_idxs]
    centroid = seed_mat.mean(axis=0)
    if not sparse.issparse(centroid):
        centroid = np.asarray(centroid)
        centroid = sparse.csr_matrix(centroid)
    scores = (centroid @ X.T).toarray().ravel()
    for si in seed_idxs:
        scores[si] = -1.0
    best = np.argpartition(-scores, range(min(k, len(scores))))[:k]
    best = best[np.argsort(-scores[best])]
    items = []
    for i in best:
        row = df.iloc[int(i)]
        items.append({
            "id": row["id"],
            "title": row["title"] if isinstance(row["title"], str) else "",
            "score": float(scores[i]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
        })
    return {"seed_count": len(seed_idxs), "results": items}


@app.get("/recommend/sbert", response_model=RecommendResponse)
def recommend_sbert(query: str = Query(...), k: int = Query(10, ge=1, le=50)):
    df = load_df()
    emb = load_sbert()
    seed_idxs = find_by_title(df, query, k=5)
    if not seed_idxs:
        return {"seed_count": 0, "results": []}
    seed_vec = emb[seed_idxs].mean(axis=0, keepdims=True)
    scores = (emb @ seed_vec.T).ravel()
    for si in seed_idxs:
        scores[si] = -1.0
    best = np.argpartition(-scores, range(min(k, len(scores))))[:k]
    best = best[np.argsort(-scores[best])]
    items = []
    for i in best:
        row = df.iloc[int(i)]
        items.append({
            "id": row["id"],
            "title": row["title"] if isinstance(row["title"], str) else "",
            "score": float(scores[i]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
        })
    return {"seed_count": len(seed_idxs), "results": items}


# Serve frontend assets
app.mount("/static", StaticFiles(directory="frontend", html=False), name="static")


@app.get("/", response_class=HTMLResponse)
def root_page():
    index_path = os.path.join("frontend", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    # Adjust static paths
    html = html.replace("/static/", "/static/")
    return HTMLResponse(html)
