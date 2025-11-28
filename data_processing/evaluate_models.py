import argparse
import json
import random
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import re


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def dcg(relevances: List[float]) -> float:
    value = 0.0
    for i, rel in enumerate(relevances, start=1):
        value += (2**rel - 1) / np.log2(i + 1)
    return value


def ndcg_at_k(relevances: List[float], k: int) -> float:
    topk = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(topk) / denom


def recall_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> float:
    topk = relevances[:k]
    num_rel = sum(1 for r in relevances if r >= threshold)
    if num_rel == 0:
        return 0.0
    got = sum(1 for r in topk if r >= threshold)
    return got / float(num_rel)


def build_corpus_row(row: pd.Series) -> str:
    title = row.get("title") or ""
    desc = row.get("description") or ""
    tags = row.get("tags") or []
    tags_text = " ".join(f"tag_{t}" for t in tags)
    return " \n ".join([str(title), str(desc), tags_text]).strip()


def normalize_tags_value(value) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list) or isinstance(value, tuple):
        seq = list(value)
    elif isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[;,|]", value) if p.strip()]
        seq = parts if parts else [value]
    else:
        return []
    out = []
    for t in seq:
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


def eval_tfidf(df: pd.DataFrame, vectorizer, X: sparse.csr_matrix, seed_idx: int, top_k: int) -> List[int]:
    query_vec = X[seed_idx]
    scores = cosine_similarity(query_vec, X).ravel()
    scores[seed_idx] = -1.0
    best = np.argpartition(-scores, range(top_k))[:top_k]
    best = best[np.argsort(-scores[best])]
    return best.tolist()


def eval_sbert(embeddings: np.ndarray, seed_idx: int, top_k: int) -> List[int]:
    seed = embeddings[seed_idx]
    scores = embeddings @ seed
    scores[seed_idx] = -1.0
    best = np.argpartition(-scores, range(top_k))[:top_k]
    best = best[np.argsort(-scores[best])]
    return best.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF vs SBERT using tag-overlap relevance")
    parser.add_argument("--data", default="mangadex_clean.parquet")
    parser.add_argument("--tfidf_model", default="models/tfidf_vectorizer.joblib")
    parser.add_argument("--tfidf_matrix", default="models/tfidf_matrix.npz")
    parser.add_argument("--sbert_dir", default="models_sbert")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    df = pd.read_parquet(args.data)
    # ensure tags are lists
    df["tags"] = df["tags"].apply(normalize_tags_value)

    # Load TF-IDF artifacts
    vectorizer = joblib.load(args.tfidf_model)
    X = sparse.load_npz(args.tfidf_matrix).tocsr()

    # Load SBERT embeddings
    emb = np.load(f"{args.sbert_dir}/embeddings.npy") if args.sbert_dir else None

    # Candidate seeds with at least 1 tag
    candidates = [i for i, t in enumerate(df["tags"]) if isinstance(t, list) and len(t) > 0]
    if len(candidates) == 0:
        raise SystemExit("No items with tags found for evaluation.")
    rng.shuffle(candidates)
    seeds = candidates[: min(args.sample_size, len(candidates))]

    k = args.k
    tfidf_recalls, tfidf_ndcgs = [], []
    sbert_recalls, sbert_ndcgs = [], []

    for idx in seeds:
        seed_tags = set(df.iloc[idx]["tags"]) if isinstance(df.iloc[idx]["tags"], list) else set()
        # graded relevance by Jaccard over tags
        relevances: Dict[int, float] = {}

        # get rec lists
        tfidf_list = eval_tfidf(df, vectorizer, X, idx, top_k=k)
        sbert_list = eval_sbert(emb, idx, top_k=k) if emb is not None else []

        # compute per-list relevances
        def compute_rels(order: List[int]) -> List[float]:
            rels: List[float] = []
            for j in order:
                cand_tags = set(df.iloc[j]["tags"]) if isinstance(df.iloc[j]["tags"], list) else set()
                rels.append(jaccard(seed_tags, cand_tags))
            return rels

        tfidf_rels = compute_rels(tfidf_list)
        sbert_rels = compute_rels(sbert_list)

        tfidf_recalls.append(recall_at_k(tfidf_rels, k=k, threshold=0.5))
        tfidf_ndcgs.append(ndcg_at_k(tfidf_rels, k=k))

        if sbert_list:
            sbert_recalls.append(recall_at_k(sbert_rels, k=k, threshold=0.5))
            sbert_ndcgs.append(ndcg_at_k(sbert_rels, k=k))

    def mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    results = {
        "k": k,
        "num_seeds": len(seeds),
        "tfidf": {"recall@k": mean(tfidf_recalls), "ndcg@k": mean(tfidf_ndcgs)},
        "sbert": {"recall@k": mean(sbert_recalls), "ndcg@k": mean(sbert_ndcgs)},
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


