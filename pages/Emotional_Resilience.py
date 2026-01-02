import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Emotional Resilience Analysis",
    layout="wide"
)

# ======================================
# TITLE & OBJECTIVE
# ======================================
st.title("Emotional Resilience and Personal Development")

st.markdown("""
**Objective:**  
To investigate the relationship between emotional resilience and personal development attributes,
including motivation, adaptability, emotional control, task persistence, and teamwork skills.
""")

# ======================================
# LOAD CLEANED DATASET (FINAL)
# ======================================
DATA_URL = "https://raw.githubusercontent.com/nhusna01/SSES-survey-dashboard/main/Hafizah_SSES_Cleaned.csv"
df = pd.read_csv(DATA_URL)

# ======================================
# OBJECTIVE 3 VARIABLES (ALREADY CLEANED)
# ======================================
likert_cols = [
    'calm_under_pressure',
    'emotional_control',
    'adaptability',
    'self_motivation',
    'task_persistence',
    'teamwork'
]

# ======================================
# DATA OVERVIEW
# ======================================
st.markdown("## 📊 Data Overview (Objective 3)")

col1, col2, col3 = st.columns(3)
col1.metric("Total Respondents", len(df))
col2.metric("Attributes Analysed", len(likert_cols))
col3.metric("Likert Scale", "1 – 5")

st.markdown("""
All variables are measured using a **5-point Likert scale**, where higher scores indicate stronger
emotional resilience or personal development characteristics.
""")

st.divider()

# ======================================
# 1️⃣ LIKERT DISTRIBUTION (STACKED BAR)
# ======================================
st.subheader("1. Distribution of Emotional Resilience Attributes")

likert_dist = df[likert_cols].apply(lambda x: x.value_counts(normalize=True)).T
likert_dist = likert_dist.reset_index().rename(columns={'index': 'Attribute'})

fig1 = px.bar(
    likert_dist,
    x="Attribute",
    y=[1, 2, 3, 4, 5],
    title="Distribution of Emotional Resilience and Personal Development Attributes",
    labels={"value": "Proportion", "variable": "Likert Scale"},
    barmode="stack"
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================
# 2️⃣ RADAR CHART (AVERAGE PROFILE)
# ======================================
st.subheader("2. Average Emotional Resilience Profile")

mean_scores = df[likert_cols].mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatterpolar(
    r=mean_scores.values.tolist() + [mean_scores.values[0]],
    theta=likert_cols + [likert_cols[0]],
    fill='toself',
    name='Average Score'
))

fig2.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    title="Average Profile of Emotional Resilience and Personal Development",
    showlegend=False
)

st.plotly_chart(fig2, use_container_width=True)

# ======================================
# 3️⃣ CORRELATION HEATMAP
# ======================================
st.subheader("3. Correlation Between Attributes")

corr_matrix = df[likert_cols].corr()

fig3 = px.imshow(
    corr_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu",
    title="Correlation Between Emotional Resilience and Personal Development Attributes"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================================
# 4️⃣ BOX PLOT (VARIABILITY)
# ======================================
st.subheader("4. Distribution and Variability")

melted = df[likert_cols].melt(
    var_name="Attribute",
    value_name="Score"
)

fig4 = px.box(
    melted,
    x="Attribute",
    y="Score",
    title="Distribution and Variability of Emotional Resilience Attributes"
)

st.plotly_chart(fig4, use_container_width=True)

# ======================================
# 5️⃣ GROUP COMPARISON (GENDER)
# ======================================
st.subheader("5. Comparison by Gender")

if "gender" in df.columns:
    gender_means = df.groupby("gender")[likert_cols].mean().reset_index()

    fig5 = px.bar(
        gender_means,
        x="gender",
        y=likert_cols,
        barmode="group",
        title="Comparison of Emotional Resilience and Personal Development by Gender"
    )

    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Gender data is not available for comparison.")

# ======================================
# SUMMARY BOX
# ======================================
st.markdown("## 🧾 Summary of Key Findings")

highest_attr = mean_scores.idxmax().replace('_', ' ').title()
lowest_attr = mean_scores.idxmin().replace('_', ' ').title()

top_corr = (
    corr_matrix.abs()
    .unstack()
    .sort_values(ascending=False)
)
top_corr = top_corr[top_corr < 1]
attr_pair = top_corr.idxmax()
corr_value = corr_matrix.loc[attr_pair]

st.markdown(f"""
- **Highest average attribute:** {highest_attr}  
- **Lowest average attribute:** {lowest_attr}  
- **Strongest relationship:** {attr_pair[0].replace('_',' ').title()} and {attr_pair[1].replace('_',' ').title()}  
  (correlation = {corr_value:.2f})  

Overall, the findings suggest that emotional resilience attributes are closely associated with
personal development skills, particularly adaptability, motivation, and task persistence.
""")
