import argparse
import json
from typing import List, Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize_tags(tags_value: Any) -> List[str]:
    """Return a list of tag strings regardless of input shape/type."""
    if tags_value is None:
        return []
    # pandas may store lists, numpy arrays, or strings
    if isinstance(tags_value, list) or isinstance(tags_value, tuple):
        seq = list(tags_value)
    elif isinstance(tags_value, np.ndarray):
        seq = tags_value.tolist()
    elif isinstance(tags_value, str):
        # attempt to split on common delimiters
        parts = [p.strip() for p in re.split(r"[;,|]", tags_value) if p.strip()]
        seq = parts if parts else [tags_value]
    else:
        return []
    cleaned = []
    for t in seq:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        cleaned.append(tt)
    return cleaned


def build_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    desc = row.get("description") or ""
    tags: List[str] = _normalize_tags(row.get("tags"))
    tags_text = " ".join(f"tag_{t}" for t in tags)
    return " \n ".join([str(title), str(desc), tags_text]).strip()


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF content-based baseline for manga recommendations")
    parser.add_argument("--data", default="mangadex_clean.parquet", help="Path to cleaned parquet dataset")
    parser.add_argument("--model_out", default="models/tfidf_vectorizer.joblib", help="Path to save vectorizer")
    parser.add_argument("--matrix_out", default="models/tfidf_matrix.npz", help="Path to save TF-IDF sparse matrix")
    parser.add_argument("--index_out", default="models/id_index.json", help="Path to save id index mapping")
    parser.add_argument("--max_features", type=int, default=200000, help="Max TF-IDF vocabulary size")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    # Build corpus
    corpus = df.apply(build_text, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True,
        strip_accents="unicode",
        norm="l2",
    )
    X = vectorizer.fit_transform(corpus)

    # Persist artifacts
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, args.model_out)
    sparse.save_npz(args.matrix_out, X)
    id_index = {int(i): idv for i, idv in enumerate(df["id"].tolist())}
    with open(args.index_out, "w", encoding="utf-8") as f:
        json.dump(id_index, f)

    print({
        "rows": X.shape[0],
        "features": X.shape[1],
        "density": float(X.nnz) / float(X.shape[0] * X.shape[1]),
    })


if __name__ == "__main__":
    main()


