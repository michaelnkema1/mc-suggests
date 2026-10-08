### MC-Suggests

A manhwa recommendation system using hybrid TF-IDF and Sentence-Transformer models with a modern web interface.

### Project structure

**Core Files:**
- `api/main.py` - FastAPI backend serving recommendations
- `frontend/` - Web interface (HTML, CSS, JavaScript)
- `mangadex_clean.parquet` - Cleaned dataset (processed from raw data)
- `models/` - Trained TF-IDF models and vectorizer
- `models_sbert/` - Sentence-Transformer embeddings
- `covers/` - Optional local cover images (not in git). Without them, covers load from MangaDex using the `cover_url` saved by the scraper

**Development Scripts:**
- `data_processing/` - Data processing scripts and raw data files
  - `preprocess_mangadex.py` - Data cleaning pipeline
  - `train_tfidf.py` - TF-IDF model training
  - `embed_sbert.py` - Sentence-Transformer embedding generation
  - `evaluate_models.py` - Model evaluation script
  - `scrape_mangadex.py` - Pulls titles, tags, covers and stats from the MangaDex API
  - `check_artifacts.py` - Verifies the dataset and model files line up row-for-row
- `recommend.py` - CLI recommendation tool
- `recommend_sbert.py` - CLI SBERT recommendation tool

### Requirements

- Python 3.10+
- Dependencies:
  - pandas, numpy, pyarrow
  - scikit-learn, scipy, joblib
  - sentence-transformers (for SBERT)
  - fastapi, uvicorn (for API)

### Quick Start

```bash
cd /home/mykecodes/Desktop/mc-suggests
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy pyarrow scikit-learn scipy joblib sentence-transformers fastapi uvicorn[standard]

# Start the web application (one command)
./run.sh
# or using Makefile
# make run
```

Then open: http://127.0.0.1:8001/

To stop the server:

```bash
make stop
```

### Features

- **Hybrid Recommendations**: Combines TF-IDF and Sentence-Transformer models
- **Modern Web Interface**: Colorful, responsive UI with cover images
- **Status Display**: Shows completion status (✅ Completed, 🔄 Ongoing, ⏸️ Hiatus, ❌ Cancelled)
- **Chapter Estimates**: Displays estimated chapter counts based on status
- **Cover Images**: Real cover images with fallback placeholders

### API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /recommend/hybrid?query=<title>&k=12&alpha=0.85` - Hybrid recommendations
- `GET /covers/{manga_id}` - Cover images

### Refreshing the data

The dataset comes from the official [MangaDex API](https://api.mangadex.org/docs/). To pull fresh titles and retrain everything:

```bash
pip install -r data_processing/requirements.txt
make refresh    # scrape -> preprocess -> TF-IDF -> SBERT embeddings -> consistency check
```

That needs network access to `api.mangadex.org` and `huggingface.co` (for the SBERT model download). It takes roughly 10-20 minutes, mostly scraping (MangaDex allows ~5 requests/second) and embedding on CPU.

- `make scrape` only fetches data, into `data_processing/mangadex_data.json`.
- `make retrain` rebuilds `mangadex_clean.parquet`, `models/` and `models_sbert/` from that file.
- By default the scraper takes the 6,000 most-followed titles for each original language: `ko` (manhwa), `ja` (manga) and `zh` (manhua). Change this with `python data_processing/scrape_mangadex.py --languages ko --per-language 8000` (MangaDex caps each language at 10,000).

Always re-run all of `make retrain` after changing the data. The API matches rows across the parquet, the TF-IDF matrix and the embeddings by position.

To evaluate the models:

```bash
python data_processing/evaluate_models.py --data mangadex_clean.parquet --k 10
```

### CLI Usage

```bash
# TF-IDF recommendations
python recommend.py --query "Solo Leveling" --k 10

# SBERT recommendations  
python recommend_sbert.py --query "Solo Leveling" --k 10
```

## Troubleshooting

- Port in use:
  ```bash
  ss -ltnp | grep 8001
  pkill -f 'uvicorn api.main:app'
  ```
- Server crash about tabs/spaces: ensure `api/main.py` uses consistent spaces.
- SBERT model download timeouts: retry `data_processing/embed_sbert.py` later or use TF-IDF endpoints in the meantime.

### Notes

- Test/demo files are ignored by git: `frontend/debug.html`, `frontend/test.html`, `test_images.html`.


