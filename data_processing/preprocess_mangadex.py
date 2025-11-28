import argparse
import json
import re
import unicodedata
from typing import Any, List, Optional

import numpy as np
import pandas as pd


def normalize_text(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None if text in (None, float("nan")) else text
    normalized = unicodedata.normalize("NFC", text)
    # remove markdown sections like **Links:** and after
    normalized = re.sub(r"\n?\*\*Links:\*\*[\s\S]*$", "", normalized, flags=re.IGNORECASE)
    # remove URLs
    normalized = re.sub(r"https?://\S+", "", normalized)
    # remove horizontal rules and artifacts
    normalized = normalized.replace("---", " ")
    # collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def parse_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # try JSON parse first (some CSV cells may contain JSON arrays)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                raw = parsed
            else:
                raw = re.split(r";|,", s)
        except Exception:
            raw = re.split(r";|,", s)
    else:
        return []

    cleaned = set()
    for t in raw:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        tt = tt.lower().replace(" ", "_")
        cleaned.add(tt)
    return sorted(cleaned)


def normalize_demographic(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    alias = {
        "shounen": "shounen",
        "shonen": "shounen",
        "shoujo": "shoujo",
        "seinen": "seinen",
        "josei": "josei",
    }
    return alias.get(v)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MangaDex datasets")
    parser.add_argument("--csv", default="mangadex_data.csv", help="Path to CSV dataset")
    parser.add_argument("--json", default="mangadex_data.json", help="Path to JSON dataset")
    parser.add_argument("--out", default="mangadex_clean.parquet", help="Output parquet path")
    parser.add_argument("--tags_out", default="mangadex_tags.parquet", help="Exploded tags parquet path")
    args = parser.parse_args()

    # Read inputs
    try:
        df_csv = pd.read_csv(args.csv)
    except Exception:
        df_csv = pd.DataFrame()
    try:
        df_json = pd.read_json(args.json)
    except Exception:
        df_json = pd.DataFrame()

    frames = [d for d in [df_csv, df_json] if not d.empty]
    if not frames:
        raise SystemExit("No input data could be loaded.")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure expected columns exist
    for col in ["id", "title", "description", "tags", "demographic", "rating", "follows", "status", "content_rating", "year"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Deduplicate by id
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Normalize text fields
    df["title_raw"] = df["title"]
    df["title"] = df["title"].apply(normalize_text)
    df["title_lc"] = df["title"].str.lower()

    df["description_raw"] = df["description"]
    df["description"] = df["description"].apply(normalize_text)

    # Tags parsing and normalization
    df["tags"] = df["tags"].apply(parse_tags)

    # Demographic normalization
    df["demographic"] = df["demographic"].apply(normalize_demographic)

    # Enum normalizations (status, content_rating)
    def norm_enum(x: Any) -> Optional[str]:
        if not isinstance(x, str):
            return None
        return x.strip().lower()

    df["status"] = df["status"].apply(norm_enum)
    df["content_rating"] = df["content_rating"].apply(norm_enum)

    # Numeric fields
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(0, 10)
    df["follows"] = pd.to_numeric(df["follows"], errors="coerce")
    df["follows"] = df["follows"].fillna(0).astype("int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.loc[(df["year"] < 1900) | (df["year"] > 2100), "year"] = pd.NA

    # Remove items with too-short descriptions for text models
    df = df[df["description"].fillna("").str.len() >= 30]

    # Engineered features
    df["follows_log1p"] = np.log1p(df["follows"].astype(float))
    df["description_len"] = df["description"].fillna("").str.len()

    # Save main parquet
    df.to_parquet(args.out, index=False)

    # Explode tags for downstream modeling (tag table)
    tags_exploded = (
        df[["id", "tags"]]
        .explode("tags")
        .dropna()
        .rename(columns={"tags": "tag"})
        .reset_index(drop=True)
    )
    tags_exploded.to_parquet(args.tags_out, index=False)

    # Print brief summary
    print({
        "rows": int(df.shape[0]),
        "unique_ids": int(df["id"].nunique()),
        "avg_tags_per_item": float(df["tags"].apply(len).mean() if not df.empty else 0.0),
        "min_year": int(df["year"].dropna().min()) if df["year"].notna().any() else None,
        "max_year": int(df["year"].dropna().max()) if df["year"].notna().any() else None,
    })


if __name__ == "__main__":
    main()


