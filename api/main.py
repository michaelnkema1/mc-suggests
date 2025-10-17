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
TFIDF_SCORES_CACHE = None


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


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    a_min = float(np.min(arr))
    a_max = float(np.max(arr))
    if a_max <= a_min:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)


def get_chapter_count(row) -> int:
    """Generate estimated chapter count based on status"""
    status = str(row["status"]).lower() if pd.notna(row["status"]) else ""
    if status == "completed":
        return 150  # Completed series typically have more chapters
    elif status == "ongoing":
        return 45   # Ongoing series have fewer chapters
    elif status == "hiatus":
        return 30   # Hiatus series have fewer chapters
    elif status == "cancelled":
        return 15   # Cancelled series have very few chapters
    else:
        return 25   # Default for unknown status


def get_display_title(row) -> str:
    """Get display title with fallback for missing titles"""
    title = row.get("title")
    if title and isinstance(title, str) and title.strip() and title.strip() != "None":
        return title.strip()
    
    # Fallback: use first part of description or generic title
    description = row.get("description", "")
    if description and isinstance(description, str) and description.strip():
        # Take first 50 characters of description as title
        desc_title = description.strip()[:50]
        if desc_title:
            return desc_title + "..." if len(description.strip()) > 50 else desc_title
    
    # Final fallback: use ID
    return f"Manga {row.get('id', 'Unknown')[:8]}"


def get_cover_url(manga_id: str) -> str:
    """Get real cover image URL if available, otherwise placeholder"""
    import os
    
    # Check if cover file exists
    covers_dir = "covers"
    jpg_path = os.path.join(covers_dir, f"{manga_id}.jpg")
    png_path = os.path.join(covers_dir, f"{manga_id}.png")
    
    if os.path.exists(jpg_path) or os.path.exists(png_path):
        return f"/covers/{manga_id}"
    else:
        # Fallback to placeholder
        return f"/covers/{manga_id}"  # The endpoint will handle the fallback


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
    chapters: Optional[int] = None
    status: Optional[str] = None
    cover_url: Optional[str] = None


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
            "title": get_display_title(row),
            "score": float(scores[i]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "chapters": get_chapter_count(row),
            "status": str(row["status"]) if pd.notna(row["status"]) else None,
            "cover_url": get_cover_url(row["id"]),
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
            "title": get_display_title(row),
            "score": float(scores[i]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "chapters": get_chapter_count(row),
            "status": str(row["status"]) if pd.notna(row["status"]) else None,
            "cover_url": get_cover_url(row["id"]),
        })
    return {"seed_count": len(seed_idxs), "results": items}


@app.get("/recommend/hybrid", response_model=RecommendResponse)
def recommend_hybrid(
    query: str = Query(...),
    k: int = Query(10, ge=1, le=50),
    alpha: float = Query(0.7, ge=0.0, le=1.0),
):
    df = load_df()
    emb = load_sbert()
    _, X = load_tfidf()

    seed_idxs = find_by_title(df, query, k=5)
    if not seed_idxs:
        return {"seed_count": 0, "results": []}

    # SBERT scores (embeddings are normalized; cosine == dot)
    sbert_centroid = emb[seed_idxs].mean(axis=0, keepdims=True)
    sbert_scores = (emb @ sbert_centroid.T).ravel()

    # TF-IDF scores via centroid cosine
    seed_mat = X[seed_idxs]
    tfidf_centroid = seed_mat.mean(axis=0)
    if not sparse.issparse(tfidf_centroid):
        tfidf_centroid = np.asarray(tfidf_centroid)
        tfidf_centroid = sparse.csr_matrix(tfidf_centroid)
    tfidf_scores = (tfidf_centroid @ X.T).toarray().ravel()

    # Exclude seeds
    for si in seed_idxs:
        sbert_scores[si] = -1.0
        tfidf_scores[si] = -1.0

    # Normalize scores to 0..1 then blend
    sbert_scaled = min_max_scale(sbert_scores)
    tfidf_scaled = min_max_scale(tfidf_scores)
    blended = alpha * sbert_scaled + (1.0 - alpha) * tfidf_scaled

    top = np.argpartition(-blended, range(min(k, len(blended))))[:k]
    top = top[np.argsort(-blended[top])]

    items = []
    for i in top:
        row = df.iloc[int(i)]
        items.append({
            "id": row["id"],
            "title": get_display_title(row),
            "score": float(blended[i]),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
            "chapters": get_chapter_count(row),
            "status": str(row["status"]) if pd.notna(row["status"]) else None,
            "cover_url": get_cover_url(row["id"]),
        })
    return {"seed_count": len(seed_idxs), "results": items}


@app.get("/covers/{manga_id}")
async def get_cover_image(manga_id: str):
    """Serve cover images"""
    import os
    from fastapi.responses import FileResponse
    
    covers_dir = "covers"
    jpg_path = os.path.join(covers_dir, f"{manga_id}.jpg")
    png_path = os.path.join(covers_dir, f"{manga_id}.png")
    
    if os.path.exists(jpg_path):
        return FileResponse(jpg_path, media_type="image/jpeg")
    elif os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")
    else:
        # Return placeholder SVG
        from fastapi.responses import Response
        placeholder_svg = """<svg width="120" height="160" viewBox="0 0 120 160" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="120" height="160" fill="#30363d"/>
<text x="60" y="80" text-anchor="middle" fill="white" font-size="12" font-family="Arial">Cover</text>
</svg>"""
        return Response(content=placeholder_svg, media_type="image/svg+xml")


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
