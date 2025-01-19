import streamlit as st



st.header('This is "the" Music reommender #1')
st.write('Finde deine Lieblingsmusik')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Film suchen')
    search_query = st.text_input('Gib einen Filmtitel ein')


