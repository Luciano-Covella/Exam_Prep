import streamlit as st


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
