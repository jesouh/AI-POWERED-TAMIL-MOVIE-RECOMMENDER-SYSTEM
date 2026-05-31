import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

# --- Optional TF-IDF fallback ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(base_dir, "tamil_movies_100.csv")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # works with 'prompt' on your setup

st.set_page_config(page_title="AI Tamil Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 AI Tamil Movie Recommender (Embeddings)")

@st.cache_data
def load_movies():
    df = pd.read_csv(CSV_FILE)
    df["title"] = df["title"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce").fillna(0.0)
    df["text"] = df["title"] + " | Genres: " + df["genre"]
    return df

df = load_movies()

# Build genre list
all_genres = sorted({g.strip() for cell in df["genre"] for g in str(cell).split(",") if g.strip()})

with st.sidebar:
    st.header("Filters")
    min_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, 6.5, 0.1)

st.subheader("Pick a genre")
picked_genre = st.selectbox("Select a genre", options=["— Select —"] + all_genres, index=0)

# ----------------- Embedding backend (prompt-based) -----------------
def ollama_embed_prompt(texts: List[str]) -> np.ndarray:
    """
    Your Ollama returns embeddings when using the 'prompt' field.
    Returns an array of vectors (float32).
    """
    vecs = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        # Expect either {"embedding":[...]} or {"data":[{"embedding":[...]}]}
        if "embedding" in data and isinstance(data["embedding"], list) and data["embedding"]:
            vecs.append(data["embedding"])
        elif "data" in data and data["data"] and "embedding" in data["data"][0]:
            vecs.append(data["data"][0]["embedding"])
        else:
            # If empty embedding comes back, raise to trigger TF-IDF fallback
            raise RuntimeError(f"Empty/unknown embedding response: {data}")
    return np.array(vecs, dtype="float32")

@st.cache_resource(show_spinner=False)
def tfidf_fit(corpus: List[str]):
    vec = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vec.fit_transform(corpus)
    return vec, X

def cosine_top_k(query_vec: np.ndarray, mat: np.ndarray, k: int = 10):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims

# Prepare TF-IDF fallback
with st.spinner("Preparing catalogue (TF-IDF ready)…"):
    tfidf_vectorizer, tfidf_matrix = tfidf_fit(df["text"].tolist())

# Try to precompute catalog embeddings with prompt
@st.cache_resource(show_spinner=True)
def try_build_catalog_embeddings_with_prompt(texts: List[str]):
    try:
        return ollama_embed_prompt(texts)
    except Exception:
        return None

catalog_embeds = try_build_catalog_embeddings_with_prompt(df["text"].tolist())

if st.button("Recommend"):
    if picked_genre == "— Select —":
        st.warning("Please pick a genre.")
        st.stop()

    st.subheader(f"AI Recommendations in **{picked_genre}**")

    # Filter pool by genre + rating
    def has_genre(genres_str: str) -> bool:
        row = {g.strip().lower() for g in str(genres_str).split(",")}
        return picked_genre.lower() in row

    pool = df[(df["imdb_rating"] >= min_rating) & (df["genre"].apply(has_genre))].copy()
    if pool.empty:
        st.warning("No movies match your filters. Try a different genre or lower the rating.")
        st.stop()

    # Query text describing the user intent
    query_text = f"Tamil movies in the genre: {picked_genre}. Recommend popular and well-rated titles."

    try:
        if catalog_embeds is not None:
            # --- Embedding path (prompt) ---
            query_vec = ollama_embed_prompt([query_text])[0]
            pool_idx = pool.index.to_numpy()
            pool_vectors = catalog_embeds[pool_idx]
            top_idx, _ = cosine_top_k(query_vec, pool_vectors, k=min(10, len(pool_vectors)))
        else:
            # --- TF-IDF fallback ---
            q_vec = tfidf_vectorizer.transform([query_text])
            pool_idx = pool.index.to_numpy()
            pool_matrix = tfidf_matrix[pool_idx]
            sims = cosine_similarity(q_vec, pool_matrix)[0]
            top_idx = np.argsort(-sims)[:min(10, len(sims))]
    except Exception:
        # Final safety net: TF-IDF
        q_vec = tfidf_vectorizer.transform([query_text])
        pool_idx = pool.index.to_numpy()
        pool_matrix = tfidf_matrix[pool_idx]
        sims = cosine_similarity(q_vec, pool_matrix)[0]
        top_idx = np.argsort(-sims)[:min(10, len(sims))]

    recs = pool.iloc[top_idx][["title", "genre", "imdb_rating"]].reset_index(drop=True)
    st.dataframe(recs, use_container_width=True)

st.markdown("---")
st.caption("Embeddings via Ollama (prompt-based). Falls back to TF-IDF if needed.")
