from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import time
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app) 
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
print("[startup] Loading recommendation model...")
t0 = time.time()
try:
    with open(os.path.join(MODEL_DIR, "cosine_sim.pkl"),   "rb") as f:
        cosine_sim   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "rb") as f:
        title_to_idx = pickle.load(f)
    movies_df = pd.read_csv(os.path.join(MODEL_DIR, "movies_df.csv"))
    print(f"[startup] Model loaded in {time.time()-t0:.2f}s  ({len(movies_df)} movies)")
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"[startup] WARNING: Model files missing — run train_model.py first!\n  {e}")
    cosine_sim   = None
    title_to_idx = {}
    movies_df    = pd.DataFrame()
    MODEL_READY  = False
def find_recommendations(movie_title: str, top_n: int = 6):
    """
    Looks up the movie in our database and returns the top-N
    most similar titles based on cosine similarity of content features.
    Returns: (list_of_recs, input_movie_info) or raises ValueError
    """
    query = movie_title.strip().lower()
    matched_key = None
    if query in title_to_idx:
        matched_key = query
    else:
        candidates = [t for t in title_to_idx if query in t]
        if candidates:
            matched_key = min(candidates, key=len)
    if matched_key is None:
        raise ValueError(f"Movie '{movie_title}' not found. Try a different title.")
    idx = title_to_idx[matched_key]
    sim_row    = cosine_sim[idx]
    all_scores = list(enumerate(sim_row))
    all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
    top_scores = [s for s in all_scores if s[0] != idx][:top_n]
    recommendations = []
    for movie_idx, sim_score in top_scores:
        row = movies_df.iloc[movie_idx]
        recommendations.append({
            "title"       : str(row["title"]),
            "genres"      : str(row["genres"]).replace(" ", ", ").title(),
            "director"    : str(row["director"]).title(),
            "year"        : int(row["year"]),
            "rating"      : float(row["rating"]),
            "overview"    : str(row["overview"]),
            "similarity"  : round(float(sim_score), 3),
        })
    src = movies_df.iloc[idx]
    input_info = {
        "title"   : str(src["title"]),
        "genres"  : str(src["genres"]).replace(" ", ", ").title(),
        "director": str(src["director"]).title(),
        "year"    : int(src["year"]),
        "rating"  : float(src["rating"]),
    }
    return recommendations, input_info
@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Postman test: GET /health → should return {"status": "ok"}
    """
    return jsonify({
        "status"      : "ok",
        "model_ready" : MODEL_READY,
        "movies_count": len(movies_df),
        "message"     : "Recommendation server is running"
    }), 200
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Main recommendation endpoint.
    Request body (JSON):
      {
        "movie": "Inception",
        "top_n": 6          (optional, default 6, max 10)
      }
    Response (JSON):
      {
        "success": true,
        "input_movie": { title, genres, director, year, rating },
        "recommendations": [ { title, genres, director, year, rating,
                                overview, similarity }, ... ]
      }
    Error response:
      {
        "success": false,
        "error": "Movie not found..."
      }
    """
    if not MODEL_READY:
        return jsonify({
            "success": False,
            "error"  : "Model not loaded. Run train_model.py first."
        }), 503
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "error"  : "Request body must be valid JSON with a 'movie' field."
        }), 400
    movie_title = data.get("movie", "").strip()
    if not movie_title:
        return jsonify({
            "success": False,
            "error"  : "The 'movie' field cannot be empty."
        }), 400
    top_n = int(data.get("top_n", 6))
    top_n = max(1, min(top_n, 10))   # clamp between 1 and 10
    try:
        recs, input_info = find_recommendations(movie_title, top_n)
        return jsonify({
            "success"        : True,
            "input_movie"    : input_info,
            "recommendations": recs,
            "count"          : len(recs),
        }), 200
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error"  : str(ve)
        }), 404
    except Exception as ex:
        # Never crash on unexpected errors — return a clean message
        return jsonify({
            "success": False,
            "error"  : f"An unexpected error occurred: {str(ex)}"
        }), 500
@app.route("/movies", methods=["GET"])
def list_movies():
    """
    Returns all movies in the database.
    Useful for autocomplete and Postman testing.

    Optional query param: ?genre=action  (filter by genre keyword)
    """
    if not MODEL_READY:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    genre_filter = request.args.get("genre", "").strip().lower()
    df_copy = movies_df.copy()
    if genre_filter:
        df_copy = df_copy[df_copy["genres"].str.contains(genre_filter, case=False)]
    movie_list = []
    for _, row in df_copy.iterrows():
        movie_list.append({
            "title"   : str(row["title"]),
            "genres"  : str(row["genres"]).replace(" ", ", ").title(),
            "director": str(row["director"]).title(),
            "year"    : int(row["year"]),
            "rating"  : float(row["rating"]),
        })
    return jsonify({
        "success": True,
        "count"  : len(movie_list),
        "movies" : movie_list
    }), 200
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed on this endpoint"}), 405
@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
