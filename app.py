import os
from html import escape
from pathlib import Path

import joblib
import sklearn
import spotipy
import streamlit as st
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from train_model import train

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Mood Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; }
    .title {
        font-size: 50px; font-weight: bold; color: white;
        text-align: center; margin-top: 20px;
    }
    .subtitle {
        color: #cbd5e1; text-align: center;
        font-size: 20px; margin-bottom: 30px;
    }
    .song-card {
        background-color: #1e293b; padding: 20px;
        border-radius: 15px; margin-bottom: 20px;
    }
    .song-card img { width: 100%; max-width: 300px; border-radius: 10px; }
    .song-title { color: white; font-size: 22px; font-weight: bold; margin-top: 10px; }
    .artist { color: #cbd5e1; font-size: 16px; }
    .listen { color: #1DB954 !important; font-weight: bold; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# CREDENTIALS (.env locally, st.secrets when deployed)
# =====================================================

load_dotenv()


def get_secret(key):
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return None


CLIENT_ID = get_secret("Client_ID")
CLIENT_SECRET = get_secret("Client_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    st.error(
        "Spotify credentials are missing. Set SPOTIFY_CLIENT_ID and "
        "SPOTIFY_CLIENT_SECRET in a local .env file, or in your "
        "hosting platform's secrets settings."
    )
    st.stop()

# =====================================================
# CACHED RESOURCES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mood_model.joblib"
MIN_CONFIDENCE = 0.5  # below this, ask the user to be more descriptive


@st.cache_resource
def load_model():
    """Load the saved pipeline; retrain in memory if it's missing or was
    built with a different scikit-learn version (training takes <1s)."""
    try:
        saved = joblib.load(MODEL_PATH)
        if saved.get("sklearn") == sklearn.__version__:
            return saved["pipeline"]
    except Exception:
        pass
    return train()


@st.cache_resource
def get_spotify():
    auth = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        cache_handler=spotipy.MemoryCacheHandler(),  # no token file on disk
    )
    return spotipy.Spotify(auth_manager=auth)


@st.cache_data(ttl=3600, show_spinner=False)
def search_tracks(query):
    results = get_spotify().search(q=query, type="track", limit=10)
    return results.get("tracks", {}).get("items", [])


model = load_model()

# =====================================================
# MOOD KEYWORDS
# =====================================================

MOOD_KEYWORDS = {
    "happy": "happy upbeat energetic",
    "sad": "sad emotional calm",
    "angry": "rock intense powerful",
    "calm": "peaceful relaxing chill",
    "romantic": "love romantic soft",
    "energetic": "workout motivational energetic",
}

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<p class="title">🎵 Mood-Based Song Recommendation</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Find songs based on your mood, language, '
    "genre and favorite artist</p>",
    unsafe_allow_html=True,
)

# =====================================================
# USER INPUTS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    user_input = st.text_input(
        "How are you feeling today?", placeholder="Example: I feel lonely"
    )
    language = st.selectbox(
        "Choose Language", ["English", "Japanese", "Korean", "Hindi"]
    )

with col2:
    genre = st.selectbox(
        "Choose Genre",
        ["Pop", "Rock", "Hip Hop", "Lofi", "Jazz", "Classical", "Anime", "K-Pop"],
    )
    favorite_artist = st.text_input(
        "Favorite Artist (Optional)", placeholder="Example: BTS"
    )

# =====================================================
# RECOMMEND SONGS
# =====================================================

if st.button("🎧 Recommend Songs", use_container_width=True):

    if not user_input.strip():
        st.warning("Please enter your mood first.")
        st.stop()

    # Predict mood
    try:
        probs = model.predict_proba([user_input])[0]
        best = probs.argmax()
        predicted_mood = str(model.classes_[best]).lower().strip()
        confidence = float(probs[best])
    except Exception as e:
        st.error(f"Mood prediction failed: {e}")
        st.stop()

    if confidence < MIN_CONFIDENCE:
        st.warning(
            "I couldn't read that mood confidently. Try describing it a "
            "bit more, e.g. 'I feel lonely and a little heartbroken'."
        )
        st.stop()

    if predicted_mood not in MOOD_KEYWORDS:
        st.warning(
            f"The model predicted '{predicted_mood}', but this mood is not "
            "configured. Available moods: " + ", ".join(MOOD_KEYWORDS)
        )
        st.stop()

    st.success(f"Detected Mood: {predicted_mood.upper()} ({confidence:.0%} confident)")

    # Build query
    query_parts = [MOOD_KEYWORDS[predicted_mood], genre, language]
    if favorite_artist.strip():
        query_parts.append(favorite_artist.strip())
    query = " ".join(query_parts)

    st.write("### 🔍 Spotify Search Query")
    st.code(query)

    # Search Spotify
    try:
        with st.spinner("Finding songs..."):
            tracks = search_tracks(query)
    except Exception as e:
        st.error(f"Spotify search failed. Please try again shortly. ({e})")
        st.stop()

    if not tracks:
        st.error("No songs found. Try another mood, language, genre or artist.")
        st.stop()

    st.write("## 🎶 Recommended Songs")
    cols = st.columns(2)

    for index, track in enumerate(tracks):
        with cols[index % 2]:
            song = escape(track.get("name", "Unknown Song"))
            artists = track.get("artists", [])
            artist = escape(
                artists[0].get("name", "Unknown Artist") if artists else "Unknown Artist"
            )
            url = track.get("external_urls", {}).get("spotify")
            images = track.get("album", {}).get("images", [])

            img_html = (
                f'<img src="{escape(images[0]["url"], quote=True)}" alt="Album art">'
                if images
                else ""
            )
            link_html = (
                f'<div style="margin-top:10px"><a class="listen" '
                f'href="{escape(url, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">▶ Listen on Spotify</a></div>'
                if url
                else ""
            )

            # Whole card rendered in ONE block so the HTML nests correctly
            st.markdown(
                f"""
                <div class="song-card">
                    {img_html}
                    <div class="song-title">🎵 {song}</div>
                    <div class="artist">🎤 {artist}</div>
                    {link_html}
                </div>
                """,
                unsafe_allow_html=True,
            )