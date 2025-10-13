import argparse
import os
import re
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_artifacts(data_path: str, out_dir: str):
    df = pd.read_parquet(data_path)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    ids_titles = pd.read_csv(os.path.join(out_dir, "id_title.csv"))
    with open(os.path.join(out_dir, "model.txt"), "r", encoding="utf-8") as f:
        model_name = f.read().strip()
    embeddings = np.load(emb_path)
    model = SentenceTransformer(model_name)
    return df, embeddings, ids_titles, model


def build_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    desc = row.get("description") or ""
    tags = row.get("tags") or []
    tags_text = " ".join(f"tag_{t}" for t in tags)
    return " \n ".join([str(title), str(desc), tags_text]).strip()


def find_by_title(df: pd.DataFrame, query: str, k: int = 1) -> List[int]:
    if not query:
        return []
    q = query.strip().lower()
    mask = df["title_lc"].fillna("").str.contains(re.escape(q))
    idxs = np.where(mask.values)[0].tolist()
    if not idxs and q:
        exact = np.where(df["title_lc"].fillna("").values == q)[0].tolist()
        idxs = exact
    return idxs[:k]


def recommend(embeddings: np.ndarray, seed_idxs: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
    if not seed_idxs:
        return []
    seed_vec = embeddings[seed_idxs].mean(axis=0, keepdims=True)
    # embeddings are normalized; cosine = dot product
    scores = embeddings @ seed_vec.T
    scores = scores.ravel()
    for si in seed_idxs:
        scores[si] = -1.0
    best = np.argpartition(-scores, range(top_k))[:top_k]
    best = best[np.argsort(-scores[best])]
    return [(int(i), float(scores[i])) for i in best]


def main():
    parser = argparse.ArgumentParser(description="SBERT-based recommendations")
    parser.add_argument("--data", default="mangadex_clean.parquet")
    parser.add_argument("--out_dir", default="models_sbert")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    df, embeddings, ids_titles, model = load_artifacts(args.data, args.out_dir)
    seed_idxs = find_by_title(df, args.query, k=5)
    recs = recommend(embeddings, seed_idxs, top_k=args.k)
    out = []
    for idx, score in recs:
        row = df.iloc[idx]
        out.append({
            "id": row["id"],
            "title": row["title"],
            "score": round(score, 6),
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": float(row["rating"]) if pd.notna(row["rating"]) else None,
        })
    print(json.dumps({"seed_count": len(seed_idxs), "results": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


