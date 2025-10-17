# Project Structure Overview

## 🎯 **What This Project Does**
A manhwa recommendation system that suggests similar titles based on content similarity using hybrid TF-IDF and Sentence-Transformer models.

## 📁 **Core Files (Essential)**
```
mc-suggests/
├── api/main.py              # FastAPI backend server
├── frontend/                 # Web interface
│   ├── index.html           # Main HTML page
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic
├── mangadex_clean.parquet   # Clean dataset (8,706 manhwas)
├── models/                  # TF-IDF models
│   ├── tfidf_vectorizer.joblib
│   ├── tfidf_matrix.npz
│   └── id_index.json
├── models_sbert/            # Sentence-Transformer embeddings
│   ├── embeddings.npy
│   └── id_title.csv
├── covers/                  # Cover images (1,000+ files)
└── README.md                # Main documentation
```

## 🛠️ **Development Scripts (Optional)**
```
├── preprocess_mangadex.py   # Data cleaning pipeline
├── train_tfidf.py          # Train TF-IDF model
├── embed_sbert.py          # Generate SBERT embeddings
├── recommend.py             # CLI TF-IDF recommendations
├── recommend_sbert.py       # CLI SBERT recommendations
└── evaluate_models.py      # Model evaluation
```

## 🚀 **Quick Start**
1. Install dependencies: `pip install pandas numpy scikit-learn sentence-transformers fastapi uvicorn`
2. Start server: `uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload`
3. Open: http://127.0.0.1:8001/

## 📊 **Data Flow**
1. **Raw Data** → `preprocess_mangadex.py` → **Clean Data** (`mangadex_clean.parquet`)
2. **Clean Data** → `train_tfidf.py` → **TF-IDF Models** (`models/`)
3. **Clean Data** → `embed_sbert.py` → **SBERT Embeddings** (`models_sbert/`)
4. **Models + Data** → `api/main.py` → **Web Interface**

## 🎨 **Features**
- **Hybrid Recommendations**: Combines TF-IDF + SBERT (85% SBERT weight)
- **Status Display**: ✅ Completed, 🔄 Ongoing, ⏸️ Hiatus, ❌ Cancelled
- **Chapter Estimates**: Based on status (completed: 150, ongoing: 45, etc.)
- **Cover Images**: Real covers with SVG placeholders
- **Modern UI**: Colorful gradient design with glass effects

## 🔧 **Configuration**
- **Port**: 8001 (configurable)
- **Alpha**: 0.85 (SBERT weight in hybrid model)
- **K**: 12 (default recommendations)
- **Model**: all-MiniLM-L6-v2 (384-dim embeddings)

## 📈 **Performance**
- **Dataset**: 8,706 manhwas
- **TF-IDF**: ~100ms response time
- **SBERT**: ~200ms response time
- **Hybrid**: ~250ms response time
- **Cover Images**: ~50ms per image
