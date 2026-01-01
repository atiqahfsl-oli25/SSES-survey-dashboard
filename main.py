import streamlit as st
import plotly.express as px
import pandas as pd
import os
import base64
from pathlib import Path
from sklearn.cluster import KMeans
from preprocess import load_data

# ======================================
# PAGE CONFIG (MUST BE FIRST)
# ======================================
st.set_page_config(
    page_title="SSES Survey Dashboard",
    layout="wide"
)

# ======================================
# BACKGROUND IMAGE FUNCTION
# ======================================
def get_base64_image(image_path):
    if not Path(image_path).exists():
        st.warning(f"Image not found: {image_path}")
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
            background-color: rgba(255, 255, 255, 0.88); /* overlay for readability */
            padding: 2rem;
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
BASE_DIR = Path(os.getcwd())
IMAGE_PATH = BASE_DIR / "assets" / "sses_background.jpg"
bg_image = get_base64_image(str(IMAGE_PATH))
set_background(bg_image)


# ======================================
# TITLE
# ======================================
st.title("SSES Survey Dashboard")
st.write("Monitoring and analyzing survey responses interactively.")

# ======================================
# LOAD DATA
# ======================================
df = load_data()

# ======================================
# SIDEBAR NAVIGATION
# ======================================
page = st.sidebar.selectbox(
    "HOMEPAGE",
    [
        "🏠 Overview",
        "👥 Demographic Analysis",
        "📊 Survey Charts", 
        "🤖 Machine Learning",
        "🎯 Emotional Resilience Analysis"
    ]
)

# ======================================
# 🏠 OVERVIEW PAGE
# ======================================
if page == "🏠 Overview":
    st.subheader("Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", len(df))
    col2.metric("Total Variables", len(df.columns))
    col3.metric("Missing Values", df.isna().sum().sum())

    st.markdown("### Data Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.write(df.describe(include="all"))

# ======================================
# 👥 DEMOGRAPHIC ANALYSIS PAGE
# ======================================
elif page == "👥 Demographic Analysis":
    st.subheader("Demographic Analysis")

    demo_col = st.selectbox(
        "Select Demographic Variable",
        options=df.columns
    )

    fig = px.pie(
        df,
        names=demo_col,
        title=f"Distribution of {demo_col}"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================
# 📊 SURVEY CHARTS PAGE
# ======================================
elif page == "📊 Survey Charts":
    st.subheader("Survey Question Analysis")

    question_col = st.selectbox(
        "Select Survey Question",
        options=df.columns
    )

    value_counts = df[question_col].value_counts().reset_index()
    value_counts.columns = [question_col, "Count"]

    fig = px.bar(
        value_counts,
        x=question_col,
        y="Count",
        text="Count",
        title=f"Response Distribution for {question_col}"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================
# 🤖 MACHINE LEARNING PAGE
# ======================================
elif page == "🤖 Machine Learning":
    st.subheader("Respondent Segmentation (K-Means)")

    st.markdown("""
    **Objective:**  
    Group respondents based on numeric survey responses using clustering.
    """)

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        st.warning("At least two numeric columns are required for clustering.")
    else:
        k = st.slider("Number of clusters (k)", 2, 6, 3)

        model = KMeans(n_clusters=k, random_state=42)
        clusters = model.fit_predict(numeric_df)

        clustered_df = numeric_df.copy()
        clustered_df["Cluster"] = clusters

        fig = px.scatter(
            clustered_df,
            x=clustered_df.columns[0],
            y=clustered_df.columns[1],
            color="Cluster",
            title="Respondent Segmentation"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Clustered Data Preview")
        st.dataframe(clustered_df.head(), use_container_width=True)

# ======================================
# 🎯 EMOTIONAL RESILIENCE ANALYSIS PAGE
# ======================================
elif page == "🎯 Emotional Resilience Analysis":
    st.subheader("Emotional Resilience and Personal Development Analysis")

    st.markdown("""
    **Objective:**  
    To investigate the relationship between emotional resilience and personal development attributes,
    including motivation, adaptability, emotional control, task persistence, and teamwork skills.
    """)

    # --------------------------------------
    # Define Objective 3 Variables (FIX)
    # --------------------------------------
    objective3_cols = [
        'calm_under_pressure',
        'emotional_control',
        'adaptability',
        'self_motivation',
        'task_persistence',
        'teamwork'
    ]

    available_cols = [c for c in objective3_cols if c in df.columns]

    if len(available_cols) < 2:
        st.error("Required variables for Emotional Resilience analysis are missing.")
        st.stop()

# ======================================
# 1. Likert Distribution
# ======================================
    st.markdown("### 1. Distribution of Emotional Resilience Attributes")

    selected_attr = st.selectbox(
        "Select Attribute",
        options=available_cols
    )

    value_counts = df[selected_attr].value_counts().reset_index()
    value_counts.columns = [selected_attr, "Count"]

    fig1 = px.bar(
        value_counts,
        x=selected_attr,
        y="Count",
        text="Count",
        title=f"Response Distribution for {selected_attr}"
    )

    st.plotly_chart(fig1, use_container_width=True)

# ======================================
# 2. Radar Chart
# ======================================
    st.markdown("### 2. Average Personal Development Profile")

    mean_scores = df[available_cols].mean().reset_index()
    mean_scores.columns = ["Attribute", "Mean Score"]

    fig2 = px.line_polar(
        mean_scores,
        r="Mean Score",
        theta="Attribute",
        line_close=True,
        title="Average Emotional Resilience and Personal Development Profile"
    )

    st.plotly_chart(fig2, use_container_width=True)
# ======================================
# 3. Correlation Heatmap
# ======================================
    st.markdown("### 3. Correlation Between Attributes")

    corr = df[available_cols].corr()

    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        title="Correlation Heatmap of Emotional Resilience Attributes"
    )

    st.plotly_chart(fig3, use_container_width=True)
# ======================================
# 4. Box Plot
# ======================================
    st.markdown("### 4. Distribution and Variability")

    melted = df[available_cols].melt(
        var_name="Attribute",
        value_name="Score"
    )

    fig4 = px.box(
        melted,
        x="Attribute",
        y="Score",
        title="Variability of Emotional Resilience and Development Attributes"
    )

    st.plotly_chart(fig4, use_container_width=True)
# ======================================
# 5. Group Comparison
# ======================================
    st.markdown("### 5. Comparison by Gender")

    if 'gender' in df.columns:
        gender_means = df.groupby('gender')[available_cols].mean().reset_index()

        fig5 = px.bar(
            gender_means,
            x="gender",
            y=available_cols,
            barmode="group",
            title="Emotional Resilience Attributes by Gender"
        )

        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Gender data is not available for group comparison.")
