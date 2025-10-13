### MC-Suggests

A minimal pipeline to preprocess MangaDex data and train a TF-IDF content-based recommender, with a simple CLI to fetch similar titles.

### Project structure

- `mangadex_data.csv`, `mangadex_data.json`: raw scraped data
- `preprocess_mangadex.py`: cleans/unifies raw data into typed, model-ready tables
- `mangadex_clean.parquet`, `mangadex_tags.parquet`: cleaned datasets (Parquet)
- `train_tfidf.py`: trains a TF-IDF model over descriptions and tags
- `models/`: saved TF-IDF vectorizer, sparse matrix, and row index mapping
- `recommend.py`: CLI to query similar titles by name

### Requirements

- Python 3.10+
- Dependencies (installed in steps below):
  - pandas, numpy, pyarrow
  - scikit-learn, scipy, joblib

### Setup

```bash
cd /home/mykecodes/Desktop/mc-suggests
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy pyarrow scikit-learn scipy joblib
```

### 1) Preprocess data

Input: `mangadex_data.csv` and/or `mangadex_data.json`. Output: cleaned Parquets.

```bash
. .venv/bin/activate
python preprocess_mangadex.py \
  --csv mangadex_data.csv \
  --json mangadex_data.json \
  --out mangadex_clean.parquet \
  --tags_out mangadex_tags.parquet
```

What it does:

- Deduplicates by `id`
- Cleans `title` and `description` (unicode normalize, strip links/markdown/artifacts)
- Parses and normalizes `tags` into a list of snake_case labels
- Normalizes `demographic`, `status`, `content_rating`
- Casts and bounds `rating`, `follows`, `year`; removes too-short descriptions
- Adds helper features: `title_lc`, `follows_log1p`, `description_len`

### 2) Train TF-IDF model

```bash
. .venv/bin/activate
python train_tfidf.py \
  --data mangadex_clean.parquet \
  --model_out models/tfidf_vectorizer.joblib \
  --matrix_out models/tfidf_matrix.npz \
  --index_out models/id_index.json
```

Outputs:

- `models/tfidf_vectorizer.joblib`: fitted `TfidfVectorizer`
- `models/tfidf_matrix.npz`: sparse TF-IDF matrix (rows align with `mangadex_clean.parquet`)
- `models/id_index.json`: row-index to `id` mapping

### 3) Get recommendations

Find titles similar to a query string (partial match against `title`):

```bash
. .venv/bin/activate
python recommend.py --query "Solo Leveling" --k 10
```

Sample output:

```json
{
  "seed_count": 2,
  "results": [
    {"id": "...", "title": "Leveling Up In An Exclusive Dungeon", "score": 0.1311, "year": 2024, "rating": 7.76},
    {"id": "...", "title": "Hardcore Leveling Warrior: Earth Game", "score": 0.1274, "year": 2023, "rating": 8.92}
  ]
}
```

### Notes & tips

- Parquet is used for speed and typed columns. If you need CSV/JSONL exports, we can add optional flags.
- `recommend.py` averages the TF-IDF vectors of all seed matches for the query and returns the top-K cosine similarities, excluding the seeds.
- To change vocabulary size, use `--max_features` in `train_tfidf.py`.

### Next steps

- Add CSV/JSONL export flags to preprocessing for Git-friendly diffs
- Improve tag normalization (synonyms, ontology)
- Add Sentence-Transformer embeddings for higher-quality semantic matches
- Expose a FastAPI endpoint for real-time recommendations


