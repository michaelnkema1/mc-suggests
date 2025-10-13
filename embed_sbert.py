import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def _normalize_tags(tags_value) -> List[str]:
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
    parser = argparse.ArgumentParser(description="Generate Sentence-Transformer embeddings for manga items")
    parser.add_argument("--data", default="mangadex_clean.parquet")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out_dir", default="models_sbert")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    corpus = df.apply(build_text, axis=1).tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        corpus,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)
    df[["id", "title"]].to_csv(os.path.join(args.out_dir, "id_title.csv"), index=False)
    with open(os.path.join(args.out_dir, "model.txt"), "w", encoding="utf-8") as f:
        f.write(args.model)

    print({
        "rows": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
    })


if __name__ == "__main__":
    main()


