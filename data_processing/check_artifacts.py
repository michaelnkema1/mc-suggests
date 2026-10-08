"""Sanity-check that the dataset and model artifacts line up row-for-row."""

import json

import numpy as np
import pandas as pd
from scipy import sparse

df = pd.read_parquet("mangadex_clean.parquet")
X = sparse.load_npz("models/tfidf_matrix.npz")
emb = np.load("models_sbert/embeddings.npy")
with open("models/id_index.json", encoding="utf-8") as f:
    id_index = json.load(f)
sbert_ids = pd.read_csv("models_sbert/id_title.csv")["id"].tolist()

ids = df["id"].tolist()
assert X.shape[0] == len(df), f"TF-IDF rows {X.shape[0]} != dataset rows {len(df)}"
assert emb.shape[0] == len(df), f"SBERT rows {emb.shape[0]} != dataset rows {len(df)}; re-run embed_sbert.py"
assert [id_index[str(i)] for i in range(len(ids))] == ids, "models/id_index.json out of sync with dataset"
assert sbert_ids == ids, "models_sbert/id_title.csv out of sync with dataset"

print({
    "rows": len(df),
    "max_year": int(df["year"].max()) if df["year"].notna().any() else None,
    "languages": df["original_language"].value_counts().to_dict() if "original_language" in df else None,
})
