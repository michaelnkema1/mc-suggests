### MC-Suggests

A manhwa recommendation system using hybrid TF-IDF and Sentence-Transformer models with a modern web interface.

### Project structure

**Core Files:**
- `api/main.py` - FastAPI backend serving recommendations
- `frontend/` - Web interface (HTML, CSS, JavaScript)
- `mangadex_clean.parquet` - Cleaned dataset (processed from raw data)
- `models/` - Trained TF-IDF models and vectorizer
- `models_sbert/` - Sentence-Transformer embeddings
- `covers/` - Cover images for manhwas

**Development Scripts:**
- `preprocess_mangadex.py` - Data cleaning pipeline
- `train_tfidf.py` - TF-IDF model training
- `embed_sbert.py` - Sentence-Transformer embedding generation
- `recommend.py` - CLI recommendation tool
- `recommend_sbert.py` - CLI SBERT recommendation tool
- `evaluate_models.py` - Model evaluation script

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

# Start the web application
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

Then open: http://127.0.0.1:8001/

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

### Development (Optional)

If you want to retrain models or process new data:

```bash
# 1. Preprocess raw data
python preprocess_mangadex.py --csv mangadex_data.csv --json mangadex_data.json --out mangadex_clean.parquet

# 2. Train TF-IDF model
python train_tfidf.py --data mangadex_clean.parquet --model_out models/tfidf_vectorizer.joblib --matrix_out models/tfidf_matrix.npz

# 3. Generate SBERT embeddings
python embed_sbert.py --data mangadex_clean.parquet --out_dir models_sbert

# 4. Evaluate models
python evaluate_models.py --data mangadex_clean.parquet --k 10
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
  ss -ltnp | grep 8000
  pkill -f 'uvicorn api.main:app'
  ```
- Server crash about tabs/spaces: ensure `api/main.py` uses consistent spaces.
- SBERT model download timeouts: retry `embed_sbert.py` later or use TF-IDF endpoints in the meantime.


