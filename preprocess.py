import pandas as pd
import streamlit as st
from config import GOOGLE_SHEET_URL

@st.cache_data(ttl=15)   # refresh every 30 seconds
def load_data():
    df = pd.read_csv(GOOGLE_SHEET_URL)
    return df

