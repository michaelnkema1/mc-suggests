import argparse
import json
import re
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def load_artifacts(data_path: str, model_path: str, matrix_path: str, index_path: str):
	# Load dataset and artifacts
	df = pd.read_parquet(data_path)
	vectorizer = joblib.load(model_path)
	X = sparse.load_npz(matrix_path)
	with open(index_path, "r", encoding="utf-8") as f:
		id_index = json.load(f)
	# invert id_index to map row idx -> id
	row_to_id = [id_index[str(i)] if isinstance(i, int) else id_index[str(i)] for i in range(len(id_index))]
	return df, vectorizer, X, row_to_id


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
		# fallback: exact lower title match if available
		exact = np.where(df["title_lc"].fillna("").values == q)[0].tolist()
		idxs = exact
	return idxs[:k]


def recommend(df: pd.DataFrame, vectorizer, X, seed_idxs: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
	if not seed_idxs:
		return []
	# Build seed vector by averaging
	seed_mat = X[seed_idxs]
	# mean over rows returns numpy.matrix; convert to CSR to satisfy sklearn
	centroid = seed_mat.mean(axis=0)
	if not sparse.issparse(centroid):
		centroid = np.asarray(centroid)
		centroid = sparse.csr_matrix(centroid)
	else:
		centroid = centroid.tocsr()
	scores = cosine_similarity(centroid, X).ravel()
	# remove seeds from candidates
	for si in seed_idxs:
		scores[si] = -1.0
	best = np.argpartition(-scores, range(top_k))[:top_k]
	best = best[np.argsort(-scores[best])]
	return [(int(i), float(scores[i])) for i in best]


def main():
	parser = argparse.ArgumentParser(description="Recommend similar manga by title query")
	parser.add_argument("--data", default="mangadex_clean.parquet")
	parser.add_argument("--model", default="models/tfidf_vectorizer.joblib")
	parser.add_argument("--matrix", default="models/tfidf_matrix.npz")
	parser.add_argument("--index", default="models/id_index.json")
	parser.add_argument("--query", required=True, help="Title substring to search as seed")
	parser.add_argument("--k", type=int, default=10)
	args = parser.parse_args()

	df, vectorizer, X, row_to_id = load_artifacts(args.data, args.model, args.matrix, args.index)
	seed_idxs = find_by_title(df, args.query, k=5)
	recs = recommend(df, vectorizer, X, seed_idxs, top_k=args.k)
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
