import streamlit as st
import pandas as pd
import os
import base64
from pathlib import Path
from preprocess import load_data

# ======================================
# PAGE CONFIG 
# ======================================
st.set_page_config(
    page_title="SSES Survey Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================================
# NAVIGATION MENU DEFINITION
# ======================================
# Ensure these filenames match your actual files in the directory
homepage = st.Page("Homepage.py", title="Homepage", icon="🏠", default=True)
demographic = st.Page("pages/Demographic_Analysis.py", title="Husna")
machine_learning = st.Page("pages/Machine_Learning.py", title="Adawiyah")
survey = st.Page("pages/Survey_Charts.py", title="Atiqah")
emotion = st.Page("pages/Emotion_Resilience.py", title="Hafizah")

pg = st.navigation(
    {
        "Menu": [homepage, demographic, machine_learning, survey, emotion]
    }
)

# ======================================
# BACKGROUND IMAGE FUNCTIONS
# ======================================
def get_base64_image(image_path):
    if not Path(image_path).exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_base64):
    if image_base64 is None:
        return
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply Background
BASE_DIR = Path(os.getcwd())
IMAGE_PATH = BASE_DIR / "assets" / "sses_background.jpg"
bg_image = get_base64_image(str(IMAGE_PATH))
set_background(bg_image)

# ======================================
# RUN NAVIGATION
# ======================================
pg.run() # Added parentheses here to actually execute the function
