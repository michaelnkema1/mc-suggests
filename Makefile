.PHONY: run dev stop scrape retrain refresh

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

run:
	./run.sh

dev:
	PORT=8001 ./run.sh

stop:
	pkill -f "uvicorn api.main:app" || true

# Pull fresh titles + stats from the MangaDex API
scrape:
	$(PYTHON) data_processing/scrape_mangadex.py --out data_processing/mangadex_data.json

# Rebuild the dataset and every model artifact from data_processing/mangadex_data.json
retrain:
	$(PYTHON) data_processing/preprocess_mangadex.py --json data_processing/mangadex_data.json --out mangadex_clean.parquet --tags_out data_processing/mangadex_tags.parquet
	$(PYTHON) data_processing/train_tfidf.py
	$(PYTHON) data_processing/embed_sbert.py
	$(PYTHON) data_processing/check_artifacts.py

refresh: scrape retrain
