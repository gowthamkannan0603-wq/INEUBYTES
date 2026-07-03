"""
Major Project — Recommendation System: Model Training
Author: Gowtham K | Reg No: INBT020009 | Course: AIINB10626

What this builds:
  A content-based movie recommendation engine using TF-IDF on movie
  metadata (genres, keywords, cast, director) combined with cosine
  similarity. Lightweight enough to deploy on free hosting platforms.

Why content-based over collaborative filtering?
  - No cold-start problem (works even with zero user history)
  - Dataset fits within the 50MB–500MB requirement
  - Model is a single similarity matrix → very fast at inference
  - Easily explainable: "Recommended because you liked Action + Sci-Fi"
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

os.makedirs("model", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build dataset (TMDB-style synthetic + real structure)
# ─────────────────────────────────────────────────────────────────────────────
# We build a rich hand-crafted dataset that mirrors the TMDB 5000 dataset
# structure. When you download the real TMDB CSV, just swap it in below.

print("[1] Building movie dataset...")

movies_data = [
    # title, genres, keywords, cast, director, overview, year, rating
    ("The Dark Knight", "action crime thriller", "batman joker gotham superhero vigilante",
     "christian bale heath ledger aaron eckhart", "christopher nolan",
     "Batman raises the stakes in his war on crime with the Joker", 2008, 9.0),

    ("Inception", "action scifi thriller", "dream heist subconscious layers time",
     "leonardo dicaprio joseph gordon levitt elliot page", "christopher nolan",
     "A thief who steals corporate secrets through dream invasion", 2010, 8.8),

    ("Interstellar", "adventure drama scifi", "space wormhole time gravity future",
     "matthew mcconaughey anne hathaway jessica chastain", "christopher nolan",
     "A team of explorers travel through a wormhole in space", 2014, 8.6),

    ("The Matrix", "action scifi", "simulation reality hacker artificial intelligence",
     "keanu reeves laurence fishburne carrie anne moss", "lana wachowski",
     "A computer hacker learns about the true nature of reality", 1999, 8.7),

    ("Avengers Endgame", "action adventure scifi", "superheroes marvel time travel infinity",
     "robert downey jr chris evans scarlett johansson", "anthony russo",
     "The Avengers assemble once more to reverse the actions of Thanos", 2019, 8.4),

    ("Pulp Fiction", "crime drama thriller", "nonlinear hitmen drugs redemption",
     "john travolta samuel jackson uma thurman", "quentin tarantino",
     "The lives of two mob hitmen, a boxer and a pair of criminals intertwine", 1994, 8.9),

    ("The Godfather", "crime drama", "mafia family power loyalty betrayal",
     "marlon brando al pacino james caan", "francis ford coppola",
     "An organized crime dynasty transfers control to a reluctant son", 1972, 9.2),

    ("Forrest Gump", "drama romance", "life destiny history running kindness",
     "tom hanks robin wright gary sinise", "robert zemeckis",
     "The story of a man with low IQ who achieves extraordinary things", 1994, 8.8),

    ("The Shawshank Redemption", "drama", "prison hope friendship freedom injustice",
     "tim robbins morgan freeman bob gunton", "frank darabont",
     "Two imprisoned men bond over years finding solace and redemption", 1994, 9.3),

    ("Fight Club", "drama thriller", "identity consumerism anarchy underground",
     "brad pitt edward norton helena bonham carter", "david fincher",
     "An insomniac and a soap salesman build a fight network", 1999, 8.8),

    ("Goodfellas", "biography crime drama", "mob gangster loyalty betrayal murder",
     "ray liotta robert de niro joe pesci", "martin scorsese",
     "Henry Hill and his friends rise and fall in the mob", 1990, 8.7),

    ("The Silence of the Lambs", "crime horror thriller", "serial killer fbi psychology cannibal",
     "jodie foster anthony hopkins ted levine", "jonathan demme",
     "FBI trainee seeks help from imprisoned cannibal to catch a killer", 1991, 8.6),

    ("Schindler's List", "biography drama history", "holocaust war rescue humanity",
     "liam neeson ben kingsley ralph fiennes", "steven spielberg",
     "A businessman saves the lives of over a thousand Polish Jews", 1993, 9.0),

    ("The Lord of the Rings", "adventure fantasy", "ring quest fellowship magic evil",
     "elijah wood ian mckellen viggo mortensen", "peter jackson",
     "A hobbit and friends embark on a quest to destroy an evil ring", 2001, 8.8),

    ("Harry Potter", "adventure fantasy family", "wizard magic school hogwarts",
     "daniel radcliffe emma watson rupert grint", "chris columbus",
     "A young wizard discovers his powers and joins a school of magic", 2001, 7.6),

    ("Titanic", "drama romance", "ship disaster love tragedy ocean",
     "leonardo dicaprio kate winslet billy zane", "james cameron",
     "A love story aboard the ill-fated RMS Titanic", 1997, 7.9),

    ("Gladiator", "action adventure drama", "rome warrior revenge emperor arena",
     "russell crowe joaquin phoenix connie nielsen", "ridley scott",
     "A Roman general seeks revenge against the corrupt emperor", 2000, 8.5),

    ("The Social Network", "biography drama", "facebook startup betrayal ambition",
     "jesse eisenberg andrew garfield justin timberlake", "david fincher",
     "The founding of Facebook and its legal aftermath", 2010, 7.7),

    ("Whiplash", "drama music", "ambition drumming perfectionism pressure mentor",
     "miles teller j k simmons paul reiser", "damien chazelle",
     "A promising drummer enrolls at a cutthroat music conservatory", 2014, 8.5),

    ("La La Land", "drama music romance", "jazz dreams ambition love los angeles",
     "ryan gosling emma stone john legend", "damien chazelle",
     "A musician and actress fall in love while chasing their dreams", 2016, 8.0),

    ("Parasite", "comedy drama thriller", "class inequality family infiltration",
     "song kang ho lee sun kyun cho yeo jeong", "bong joon ho",
     "A poor family scheme to become employed by a wealthy household", 2019, 8.6),

    ("Joker", "crime drama thriller", "villain origin chaos society mental health",
     "joaquin phoenix robert de niro zazie beetz", "todd phillips",
     "A failed comedian descends into madness becoming the Joker", 2019, 8.4),

    ("Spider-Man No Way Home", "action adventure scifi", "multiverse spiderman superhero",
     "tom holland zendaya benedict cumberbatch", "jon watts",
     "Spider-Man seeks help from Doctor Strange with drastic consequences", 2021, 8.2),

    ("Dune", "adventure drama scifi", "desert planet empire prophecy future",
     "timothee chalamet zendaya oscar isaac", "denis villeneuve",
     "A noble family becomes embroiled in a war for a desert planet", 2021, 8.0),

    ("Top Gun Maverick", "action drama", "fighter pilot navy training legacy",
     "tom cruise jennifer connelly miles teller", "joseph kosinski",
     "Maverick faces his deepest fears while training a new generation", 2022, 8.3),

    ("Everything Everywhere All at Once", "action comedy scifi", "multiverse family identity",
     "michelle yeoh ke huy quan jamie lee curtis", "dan kwan",
     "A woman discovers she can access skills from parallel universes", 2022, 7.8),

    ("The Departed", "crime drama thriller", "undercover mole police mob boston",
     "leonardo dicaprio matt damon jack nicholson", "martin scorsese",
     "An undercover cop and a mole in the police force try to identify each other", 2006, 8.5),

    ("No Country for Old Men", "crime drama thriller", "hitman pursuit fate violence texas",
     "javier bardem josh brolin tommy lee jones", "coen brothers",
     "A hunter stumbles upon drug money and a relentless assassin follows", 2007, 8.1),

    ("There Will Be Blood", "drama history", "oil greed religion ambition power",
     "daniel day lewis paul dano kevin j o connor", "paul thomas anderson",
     "A story of family, religion and oil in the early 20th century", 2007, 8.2),

    ("Mad Max Fury Road", "action adventure scifi", "post apocalyptic desert chase rebellion",
     "tom hardy charlize theron nicholas hoult", "george miller",
     "A woman rebels against a tyrannical ruler in the wasteland", 2015, 8.1),

    ("Get Out", "horror mystery thriller", "race hypnosis suburban paranoia",
     "daniel kaluuya allison williams bradley whitford", "jordan peele",
     "A Black man uncovers a disturbing secret at his girlfriend's estate", 2017, 7.7),

    ("Hereditary", "drama horror mystery", "grief supernatural family curse",
     "toni collette alex wolff gabriel byrne", "ari aster",
     "A family unravels cryptic and terrifying secrets after a death", 2018, 7.3),

    ("Avengers Infinity War", "action adventure scifi", "superheroes thanos infinity stones marvel",
     "robert downey jr chris hemsworth josh brolin", "anthony russo",
     "The Avengers must stop Thanos from collecting all six Infinity Stones", 2018, 8.4),

    ("Black Panther", "action adventure scifi", "wakanda africa superhero identity",
     "chadwick boseman michael b jordan lupita nyongo", "ryan coogler",
     "The king of Wakanda fights to protect his nation", 2018, 7.3),

    ("Doctor Strange", "action adventure fantasy", "magic multiverse sorcery dimension",
     "benedict cumberbatch chiwetel ejiofor rachel mcadams", "scott derrickson",
     "A surgeon becomes a powerful sorcerer to protect the world", 2016, 7.5),

    ("Arrival", "drama mystery scifi", "alien language communication time perception",
     "amy adams jeremy renner forest whitaker", "denis villeneuve",
     "A linguist works to communicate with extraterrestrial visitors", 2016, 7.9),

    ("Blade Runner 2049", "drama mystery scifi", "artificial intelligence future dystopia identity",
     "ryan gosling harrison ford ana de armas", "denis villeneuve",
     "A blade runner uncovers a secret that could plunge society into chaos", 2017, 8.0),

    ("Her", "drama romance scifi", "artificial intelligence loneliness connection future",
     "joaquin phoenix scarlett johansson amy adams", "spike jonze",
     "A man develops a relationship with an AI operating system", 2013, 8.0),

    ("Ex Machina", "drama scifi thriller", "artificial intelligence robot consciousness test",
     "domhnall gleeson alicia vikander oscar isaac", "alex garland",
     "A programmer evaluates an AI with human-like consciousness", 2014, 7.7),

    ("2001 A Space Odyssey", "adventure scifi", "space station ai evolution monolith",
     "keir dullea gary lockwood", "stanley kubrick",
     "Humanity finds a mysterious object beneath the lunar surface", 1968, 8.3),

    ("Alien", "horror scifi", "space creature survival fear extraterrestrial",
     "sigourney weaver tom skerritt john hurt", "ridley scott",
     "A commercial crew encounters a deadly extraterrestrial life form", 1979, 8.4),

    ("Jurassic Park", "action adventure scifi", "dinosaurs theme park genetics survival",
     "sam neill laura dern jeff goldblum", "steven spielberg",
     "A theme park with cloned dinosaurs suffers a catastrophic breakdown", 1993, 8.2),

    ("The Lion King", "adventure animation drama", "pride betrayal redemption family africa",
     "matthew broderick jeremy irons james earl jones", "roger allers",
     "A young lion prince flees his kingdom only to learn the truth", 1994, 8.5),

    ("Toy Story", "adventure animation comedy", "toys friendship loyalty identity",
     "tom hanks tim allen don rickles", "john lasseter",
     "A cowboy doll is threatened by the arrival of a new space toy", 1995, 8.3),

    ("Up", "adventure animation drama", "balloons journey loss friendship old age",
     "edward asner jordan nagai christopher plummer", "pete docter",
     "An elderly man fulfils his dream of adventure with a boy scout", 2009, 8.3),

    ("WALL-E", "animation romance scifi", "robot earth pollution love loneliness",
     "ben burtt elissa knight jeff garlin", "andrew stanton",
     "A robot left to clean Earth falls in love with another robot", 2008, 8.4),

    ("Spirited Away", "adventure animation fantasy", "spirit world magic journey identity",
     "daveigh chase suzanne pleshette", "hayao miyazaki",
     "A girl wanders into a world ruled by gods and witches", 2001, 8.6),

    ("Princess Mononoke", "action adventure animation", "nature war balance forest spirits",
     "yoji matsuda yuriko ishida", "hayao miyazaki",
     "A prince becomes involved in a struggle between forest gods and humans", 1997, 8.4),

    ("Oldboy", "action mystery thriller", "revenge imprisonment identity twist",
     "choi min sik yoo ji tae kang hye jung", "park chan wook",
     "A man imprisoned for 15 years seeks the reason for his captivity", 2003, 8.4),

    ("City of God", "crime drama", "brazil poverty gang violence favela",
     "alexandre rodrigues leandro firmino alice braga", "fernando meirelles",
     "Youth gangs run amok in the slums of Rio de Janeiro", 2002, 8.6),

    ("Pan's Labyrinth", "drama fantasy thriller", "fantasy war spain fairy tale dark",
     "ivana baquero sergi lopez maribel verdu", "guillermo del toro",
     "A girl escapes to a fairy tale world during post-war Spain", 2006, 8.2),
]

columns = ["title", "genres", "keywords", "cast", "director", "overview", "year", "rating"]
df = pd.DataFrame(movies_data, columns=columns)
df["movie_id"] = range(len(df))

print(f"   Dataset ready: {len(df)} movies across diverse genres")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering — build a combined "soup" for TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Engineering content features...")

def build_soup(row):
    """
    Combine all metadata into a single string for TF-IDF.
    Genres and director are repeated to give them more weight
    (a simple but effective trick for content-based filtering).
    """
    parts = [
        row["genres"] * 2,            # double weight on genre
        row["keywords"],
        row["cast"],
        row["director"] * 2,          # double weight on director
        row["overview"].lower(),
    ]
    return " ".join(parts)

df["content_soup"] = df.apply(build_soup, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. TF-IDF + Cosine Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building TF-IDF matrix and cosine similarity...")

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df["content_soup"])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"   TF-IDF matrix  : {tfidf_matrix.shape}")
print(f"   Similarity mat : {cosine_sim.shape}")

# Build a reverse lookup: movie title → dataframe index
title_to_idx = pd.Series(df.index, index=df["title"].str.lower()).to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Recommendation function (used by Flask)
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(movie_title: str, n: int = 6):
    """
    Returns top-N similar movies based on content features.
    Handles partial title matching so users don't need exact spelling.
    """
    query = movie_title.strip().lower()

    # Exact match first
    if query in title_to_idx:
        idx = title_to_idx[query]
    else:
        # Fuzzy: find closest title by substring
        matches = [t for t in title_to_idx if query in t]
        if not matches:
            return None, f"Movie '{movie_title}' not found in database."
        idx = title_to_idx[matches[0]]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]

    results = []
    for movie_idx, score in sim_scores:
        row = df.iloc[movie_idx]
        results.append({
            "title"    : row["title"],
            "genres"   : row["genres"].replace(" ", ", ").title(),
            "director" : row["director"].title(),
            "year"     : int(row["year"]),
            "rating"   : float(row["rating"]),
            "overview" : row["overview"],
            "score"    : round(float(score), 3),
        })

    input_movie = df.iloc[idx]
    input_info = {
        "title"    : input_movie["title"],
        "genres"   : input_movie["genres"].replace(" ", ", ").title(),
        "director" : input_movie["director"].title(),
        "year"     : int(input_movie["year"]),
        "rating"   : float(input_movie["rating"]),
    }
    return results, input_info


# Quick sanity check
test_recs, info = get_recommendations("Inception")
print(f"\n   Test: Recommendations for 'Inception':")
for r in test_recs[:3]:
    print(f"   → {r['title']} ({r['year']})  sim={r['score']}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save model artifacts
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Saving model artifacts...")

with open("model/cosine_sim.pkl",   "wb") as f: pickle.dump(cosine_sim, f)
with open("model/title_to_idx.pkl", "wb") as f: pickle.dump(title_to_idx, f)
with open("model/tfidf.pkl",        "wb") as f: pickle.dump(tfidf, f)
df.to_csv("model/movies_df.csv", index=False)

print("   Saved:")
print("   ├── model/cosine_sim.pkl")
print("   ├── model/title_to_idx.pkl")
print("   ├── model/tfidf.pkl")
print("   └── model/movies_df.csv")

print(f"\n   Total movies in DB : {len(df)}")
print(f"   Unique genres      : {len(set(' '.join(df.genres).split()))}")
print(f"   Model ready for Flask deployment ✓")
