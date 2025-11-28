# Data Processing

This directory contains all scripts and data files used for data collection, preprocessing, model training, and evaluation.

## Files

### Scripts
- `preprocess_mangadex.py` - Data cleaning and preprocessing pipeline
- `train_tfidf.py` - TF-IDF model training
- `embed_sbert.py` - Sentence-Transformer embedding generation
- `evaluate_models.py` - Model evaluation script
- `mangadex_scraper2.0.py` - Data collection script from MangaDex API

### Data Files
- `mangadex_data.csv` - Raw CSV data from MangaDex
- `mangadex_data.json` - Raw JSON data from MangaDex
- `mangadex_tags.parquet` - Exploded tags table (intermediate file)
- `mangadex_clean.parquet` - **Output**: Cleaned dataset (moved to root for API use)
- `checkpoint_*.json` - Training checkpoints
- `manhwa-covers.zip` - Cover images archive (large file, not in git)

## Usage

See the main `README.md` in the project root for usage instructions.

## Note

The processed `mangadex_clean.parquet` file is kept in the project root directory because it's required by the API at runtime.
