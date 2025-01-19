import streamlit as st
import requests

# Titel und Beschreibung
st.title("🎥 Movie Recommender App")
st.write("Finde und bewerte Filme, und erhalte personalisierte Empfehlungen!")

# Layout mit Spalten
col1, col2 = st.columns(2)

# Suchleiste
with col1:
    st.subheader("🔍 Film suchen")
    search_query = st.text_input("Gib einen Filmtitel ein:")

with col2:
    st.subheader("⭐ Deine Bewertung")
    selected_movie = st.selectbox("Wähle einen Film aus:", ["Film 1", "Film 2", "Film 3"])
    rating = st.slider("Bewerte den Film (1-5 Sterne):", 1, 5)

# Empfehlungen
st.subheader("🎯 Empfehlungen")
st.write("Hier erscheinen personalisierte Filmempfehlungen.")

# Statistiken
st.subheader("📊 Statistiken")
st.write("Visualisierungen zu Bewertungen und Genres kommen hier hin.")

dummy_movies = [
    {"title": "Inception", "genre": "Sci-Fi", "year": 2010, "rating": 8.8},
    {"title": "The Dark Knight", "genre": "Action", "year": 2008, "rating": 9.0},
    {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "rating": 8.6},
    {"title": "Parasite", "genre": "Thriller", "year": 2019, "rating": 8.6},
    {"title": "Joker", "genre": "Drama", "year": 2019, "rating": 8.4},
]

st.subheader("🎥 Verfügbare Filme")
st.write("Hier sind einige Filme, die du bewerten kannst:")
st.table(dummy_movies)

