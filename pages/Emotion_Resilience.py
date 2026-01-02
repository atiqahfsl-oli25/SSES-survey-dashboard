import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================
# PAGE TITLE & PROBLEM STATEMENT
# ======================================
st.title("Emotional Resilience and Personal Development")

st.markdown("""
**Problem Statement:** Emotional resilience is a key factor in personal and professional success. This analysis investigates the relationship between emotional resilience and key personal development attributes such as adaptability, motivation, emotional control, task persistence, and teamwork skills.
""")

st.info("""
**Objective:** To investigate the relationship between emotional resilience and personal development attributes, including motivation, adaptability, emotional control, task persistence, and teamwork skills.
""")

# ======================================
# LOAD CLEANED DATASET
# ======================================
# We use the specific cleaned CSV for this page's specialized analysis
DATA_URL = "https://raw.githubusercontent.com/nhusna01/SSES-survey-dashboard/main/Hafizah_SSES_Cleaned.csv"

@st.cache_data
def load_emotion_data():
    data = pd.read_csv(DATA_URL)
    return data

df = load_emotion_data()

# ======================================
# DATA PREPARATION
# ======================================
objective3_cols = [
    'calm_under_pressure',
    'emotional_control',
    'adaptability',
    'self_motivation',
    'task_persistence',
    'teamwork'
]

# Ensure numeric and handle missing values
df[objective3_cols] = df[objective3_cols].apply(pd.to_numeric, errors="coerce")
df[objective3_cols] = df[objective3_cols].fillna(df[objective3_cols].median())

# ======================================
# 🧾 SUMMARY OVERVIEW – LIKERT DISTRIBUTION
# ======================================
st.subheader("📌 Summary Overview")

agree_levels = [4, 5]
agree_prop = df[objective3_cols].isin(agree_levels).mean()

strongest_attr = agree_prop.idxmax()
weakest_attr = agree_prop.idxmin()
overall_resilience = agree_prop.mean()

m1, m2, m3 = st.columns(3)
m1.metric("Overall Agreement", f"{overall_resilience:.1%}")
m2.metric("Strongest Attribute", strongest_attr.replace("_", " ").title())
m3.metric("Growth Area", weakest_attr.replace("_", " ").title())

st.success(f"""
**Interpretation:** Respondents generally demonstrate positive emotional resilience. 
The highest agreement is in **{strongest_attr.replace("_", " ").title()}**, 
while **{weakest_attr.replace("_", " ").title()}** is the primary area for potential development.
""")

# ======================================
# 1️⃣ LIKERT DISTRIBUTION (STACKED BAR)
# ======================================
st.subheader("1. Attribute Distribution")

likert_counts = df[objective3_cols].apply(lambda x: x.value_counts(normalize=True)).T.reset_index().rename(columns={"index": "Attribute"})
likert_long = likert_counts.melt(id_vars="Attribute", var_name="Likert Scale", value_name="Proportion")

fig1 = px.bar(
    likert_long,
    x="Attribute",
    y="Proportion",
    color="Likert Scale",
    barmode="stack",
    color_discrete_sequence=px.colors.sequential.Viridis,
    title="Likert Distribution of Resilience Attributes"
)
st.plotly_chart(fig1, use_container_width=True)

# ======================================
# 2️⃣ RADAR CHART (AVERAGE PROFILE)
# ======================================
st.subheader("2. Average Resilience Profile")

mean_scores = df[objective3_cols].mean()
fig2 = go.Figure()
fig2.add_trace(go.Scatterpolar(
    r=mean_scores.values.tolist() + [mean_scores.values[0]],
    theta=[c.replace("_", " ").title() for c in objective3_cols] + [objective3_cols[0].replace("_", " ").title()],
    fill='toself',
    line_color='#1f77b4'
))
fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=False
)
st.plotly_chart(fig2, use_container_width=True)

# ======================================
# 3️⃣ CORRELATION HEATMAP
# ======================================
st.subheader("3. Attribute Correlations")

corr_matrix = df[objective3_cols].corr()
fig3 = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)
st.plotly_chart(fig3, use_container_width=True)

# ======================================
# 4️⃣ GROUP COMPARISON (GENDER)
# ======================================
st.subheader("4. Comparison by Gender")

if "gender" in df.columns:
    gender_means = df.groupby("gender")[objective3_cols].mean().reset_index()
    fig4 = px.bar(
        gender_means,
        x="gender",
        y=objective3_cols,
        barmode="group",
        title="Average Scores by Gender"
    )
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Gender variable not found in this specific CSV.")
