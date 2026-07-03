# Task 2 – Sentiment Analysis Using ML and DL
**Author:** Gowtham K | **Reg No:** INBT020009 | **Course ID:** AIINB10626

---

## Overview

| | Part A | Part B |
|---|---|---|
| Approach | Traditional ML | Deep Learning |
| Model 1 | Logistic Regression | Bidirectional LSTM |
| Model 2 | Linear SVM | — |
| Features | TF-IDF (unigrams + bigrams) | Learned Embeddings |
| Dataset | IMDb 50K Reviews | Same split as Part A |

---

## Dataset
- **IMDb Movie Reviews** — 50,000 reviews (25K pos / 25K neg)
- Loaded via `tensorflow_datasets`
- **Fixed split**: 80% train / 20% test with `random_state=42` (same across both parts)
- **Preprocessing**: lowercase → strip HTML → remove punctuation → remove stopwords → tokenise

---

## How to Run

```bash
pip install -r requirements.txt

# Part A — Logistic Regression + SVM
cd partA
python sentiment_ml.py

# Part B — Bidirectional LSTM (run AFTER Part A)
cd ../partB
python sentiment_lstm.py
```

> **Important**: Run Part A first — it saves the train/test split that Part B loads.

---

## Part A — ML Pipeline

```
Raw Text
  → Lowercase + strip HTML + remove punctuation
  → Remove stopwords (NLTK)
  → TF-IDF Vectoriser (30,000 features, unigrams+bigrams, sublinear_tf)
  → Logistic Regression (C=1.0, lbfgs)
  → Linear SVM (C=0.5)
```

**TF-IDF Settings:**
- `max_features=30,000`
- `ngram_range=(1, 2)` — captures phrases like "not good", "very bad"
- `sublinear_tf=True` — log-smoothed term frequency
- `min_df=3` — ignores extremely rare words

---

## Part B — LSTM Architecture

```
Token Input (max_len=300)
  → Embedding(20001, 64) + SpatialDropout(0.2)
  → Bidirectional LSTM(64, return_sequences=True) + Dropout(0.2)
  → Bidirectional LSTM(32, return_sequences=True)
  → GlobalMaxPooling1D
  → Dropout(0.3) → Dense(64, ReLU)
  → Dropout(0.2) → Dense(1, Sigmoid)
```

**Why Bidirectional LSTM?**
Text sentiment often depends on both past and future context within a sentence. "Not bad at all" needs the full sentence to understand the positive sentiment. BiLSTM reads both directions and concatenates the representations.

**Why Global Max Pooling over final hidden state?**
Long reviews (300 tokens) may have the key sentiment signal anywhere — not just at the end. GlobalMaxPool picks the most activated feature across all time steps.

---

## Preprocessing Pipeline Explained

| Step | What | Why |
|---|---|---|
| Lowercase | `text.lower()` | Normalise vocabulary |
| HTML stripping | `re.sub(r"<[^>]+>")` | IMDb reviews contain `<br />` tags |
| Punctuation removal | `re.sub(r"[^a-z\s]")` | Reduce noise |
| Stopword removal | NLTK stopwords | Remove uninformative words |
| Tokenisation (Part A) | Whitespace split → TF-IDF | Sparse feature matrix |
| Tokenisation (Part B) | Keras Tokenizer → integer seqs | Dense sequences for LSTM |

---

## Expected Output Files

```
partA/results/
  ├── partA_results.png       ← class dist + metric comparison + confusion matrices
  ├── partA_features.png      ← top 20 pos/neg TF-IDF features
  ├── partA_metrics.npy       ← LR and SVM scores (loaded by Part B)
  ├── train_texts_raw.npy     ← saved train split (reused in Part B)
  ├── test_texts_raw.npy      ← saved test split (reused in Part B)
  ├── tfidf_vectorizer.pkl
  ├── logistic_regression_model.pkl
  └── svm_model.pkl

partB/results/
  ├── partB_results.png       ← LSTM curves + confusion matrix + 3-way comparison
  └── lstm_best.keras         ← saved best LSTM weights
```

---

## Deliverables Checklist
- [x] Python code (Part A + Part B)
- [x] requirements.txt
- [x] README
- [ ] Preprocessing explanation (fill from console output)
- [ ] Architecture diagrams (from model.summary())
- [ ] Performance table (fill after running)
- [ ] Confusion matrices (auto-saved as PNG)
- [ ] Google Doc report
