import streamlit as st

def show_sidebar():
    with st.sidebar:
        st.title("📊 Portfolio Menu")


        menu = st.radio("Navigation", [
            "📁 Upload CSV",
            "📈 Portfolio Overview",
            "📉 Performance & Risk Analytics"
        ])
        return menu
