# CineMatch — AI Movie Recommendation System
**Author:** Gowtham K &nbsp;|&nbsp; **Reg No:** INBT020009 &nbsp;|&nbsp; **Course ID:** AIINB10626  
**Type:** Full-Stack AI Web Application &nbsp;|&nbsp; **Option:** A — Recommendation System

---

## What It Does
CineMatch is a content-based movie recommendation web app. A user enters any movie title, and the AI returns the 6 most similar movies based on genre, keywords, cast, director and story overview — all served through a clean, responsive web interface.

---

## How the Model Works

### Content-Based Filtering with TF-IDF + Cosine Similarity
```
Movie Metadata (genres, keywords, cast, director, overview)
    ↓
Feature Engineering → "content soup" string per movie
    ↓
TF-IDF Vectoriser (5000 features, unigrams+bigrams)
    ↓
Cosine Similarity Matrix (N × N)
    ↓
Query: user inputs movie → find row → sort by similarity → return top-N
```

**Why content-based?**
- No cold-start problem — works without user history
- Lightweight: the model is a single similarity matrix (~50KB for 50 movies)
- Fast at inference: O(1) lookup after precomputation
- Easily explainable results

**Why genres and director are double-weighted:**
Movies by the same director or in the same genre tend to feel more similar to viewers, so these fields are repeated twice in the content soup before TF-IDF encoding.

---

## Project Structure
```
major_project/
├── app.py                  ← Flask backend (API + frontend serving)
├── train_model.py          ← Model training script
├── requirements.txt
├── Procfile                ← For Render/Railway deployment
├── render.yaml             ← Render deployment config
├── postman_collection.json ← API test documentation
├── templates/
│   └── index.html          ← Full frontend (HTML + CSS + JS)
└── model/                  ← Generated after running train_model.py
    ├── cosine_sim.pkl
    ├── title_to_idx.pkl
    ├── tfidf.pkl
    └── movies_df.csv
```

---

## Setup & Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates model/ folder)
python train_model.py

# 3. Start the Flask server
python app.py
# → Open http://localhost:5000
```

---

## API Endpoints

### GET `/health`
Confirms the server is running.
```json
{
  "status": "ok",
  "model_ready": true,
  "movies_count": 50,
  "message": "Recommendation server is running"
}
```

### POST `/recommend`
Returns movie recommendations.

**Request:**
```json
{ "movie": "Inception", "top_n": 6 }
```

**Response:**
```json
{
  "success": true,
  "input_movie": {
    "title": "Inception",
    "genres": "Action, Scifi, Thriller",
    "director": "Christopher Nolan",
    "year": 2010,
    "rating": 8.8
  },
  "recommendations": [
    {
      "title": "Interstellar",
      "genres": "Adventure, Drama, Scifi",
      "director": "Christopher Nolan",
      "year": 2014,
      "rating": 8.6,
      "overview": "...",
      "similarity": 0.847
    }
  ],
  "count": 6
}
```

**Error (movie not found):**
```json
{ "success": false, "error": "Movie 'xyz' not found. Try a different title." }
```

### GET `/movies?genre=scifi`
Lists all movies, with optional genre filter.

---

## Error Handling
| Scenario | HTTP Status | Response |
|---|---|---|
| Movie not found | 404 | `{"success": false, "error": "..."}` |
| Empty input | 400 | `{"success": false, "error": "..."}` |
| Model not loaded | 503 | `{"success": false, "error": "..."}` |
| Wrong HTTP method | 405 | `{"success": false, "error": "..."}` |
| Server error | 500 | `{"success": false, "error": "..."}` |

---

## Deployment on Render (Free Tier)

1. Push the project to a **public GitHub repo**
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repo
4. Set:
   - **Build command:** `pip install -r requirements.txt && python train_model.py`
   - **Start command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120`
5. Deploy — your live URL will appear in ~3 minutes

**Cold start note:** Free Render instances sleep after 15 minutes of inactivity. First load may take 20–30 seconds.

---

## Deployment Performance Guidelines
- Model size: ~100KB (well within free hosting limits)
- Response time: < 100ms per request (cosine similarity is precomputed)
- No large files bundled in repo — model generated at build time
- 2 Gunicorn workers handle concurrent requests without crashing

---

## Postman Testing
Import `postman_collection.json` into Postman. Set the `base_url` variable to:
- Local: `http://localhost:5000`
- Live: your Render URL

Run all 8 test cases covering normal flow, partial matching, and all error cases.

---

## Security Notes
- No API keys, passwords or tokens are hardcoded
- `.env` file is used for any sensitive config (not committed to GitHub)
- All inputs are validated before processing
- CORS is enabled for frontend–backend communication

---

## Deliverables Checklist
- [x] Flask backend (`app.py`) with all required endpoints
- [x] Frontend (`templates/index.html`) — clean, usable, non-technical friendly
- [x] Model training script (`train_model.py`)
- [x] requirements.txt
- [x] Procfile + render.yaml (deployment config)
- [x] Postman collection (8 test cases)
- [x] README with setup, usage, testing and deployment instructions
- [ ] Live deployed link (fill after Render deployment)
- [ ] GitHub repository link (fill after pushing)
- [ ] Screenshots of working application
