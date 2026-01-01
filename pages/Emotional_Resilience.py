import streamlit as st
import plotly.express as px
import pandas as pd
from preprocess import load_data

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Emotional Resilience Analysis",
    layout="wide"
)

# ======================================
# LOAD DATA
# ======================================
df = load_data()

st.title("Emotional Resilience and Personal Development")

st.markdown("""
**Objective:**  
To investigate the relationship between emotional resilience and personal development attributes,
including motivation, adaptability, emotional control, task persistence, and teamwork skills.
""")

# ======================================
# OBJECTIVE 3 VARIABLE MAPPING
# (MATCH YOUR GOOGLE FORM HEADERS)
# ======================================
objective3_map = {
    "Calm Under Pressure": "I can stay calm even when under pressure.",
    "Emotional Control": "I can control my emotions when I feel angry or upset.",
    "Adaptability": "I can adapt easily to new or unexpected situations.",
    "Self Motivation": "I am motivated to improve my skills and knowledge.",
    "Task Persistence": "I finish tasks even when they are difficult.",
    "Teamwork": "I find it easy to work well with others."
}

available_map = {
    label: col for label, col in objective3_map.items()
    if col in df.columns
}

if len(available_map) < 2:
    st.error("Objective 3 variables not found in the dataset.")
    st.stop()

# Convert Likert to numeric
df[list(available_map.values())] = df[list(available_map.values())].apply(
    pd.to_numeric, errors="coerce"
)

# ======================================
# 1️⃣ LIKERT DISTRIBUTION (BAR CHART)
# ======================================
st.subheader("1. Distribution of Emotional Resilience Attributes")

selected_label = st.selectbox(
    "Select Attribute",
    options=list(available_map.keys())
)

selected_col = available_map[selected_label]

dist = df[selected_col].value_counts().reset_index()
dist.columns = ["Response", "Count"]

fig1 = px.bar(
    dist,
    x="Response",
    y="Count",
    text="Count",
    title=f"Response Distribution: {selected_label}"
)

st.plotly_chart(fig1, use_container_width=True)

# ======================================
# 2️⃣ RADAR CHART (AVERAGE PROFILE)
# ======================================
st.subheader("2. Average Emotional Resilience Profile")

mean_scores = pd.DataFrame({
    "Attribute": list(available_map.keys()),
    "Mean Score": [df[col].mean() for col in available_map.values()]
})

fig2 = px.line_polar(
    mean_scores,
    r="Mean Score",
    theta="Attribute",
    line_close=True,
    title="Average Emotional Resilience and Personal Development Profile"
)

st.plotly_chart(fig2, use_container_width=True)

# ======================================
# 3️⃣ CORRELATION HEATMAP
# ======================================
st.subheader("3. Correlation Between Attributes")

corr = df[list(available_map.values())].corr()
corr.columns = available_map.keys()
corr.index = available_map.keys()

fig3 = px.imshow(
    corr,
    text_auto=".2f",
    title="Correlation Heatmap of Emotional Resilience Attributes"
)

st.plotly_chart(fig3, use_container_width=True)

# ======================================
# 4️⃣ BOX PLOT (VARIABILITY)
# ======================================
st.subheader("4. Distribution and Variability")

melted = df[list(available_map.values())].melt(
    var_name="Attribute",
    value_name="Score"
)

reverse_map = {v: k for k, v in available_map.items()}
melted["Attribute"] = melted["Attribute"].map(reverse_map)

fig4 = px.box(
    melted,
    x="Attribute",
    y="Score",
    title="Variability of Emotional Resilience and Development Attributes"
)

st.plotly_chart(fig4, use_container_width=True)

# ======================================
# 5️⃣ GROUP COMPARISON (GENDER)
# ======================================
st.subheader("5. Comparison by Gender")

if "gender" in df.columns:
    gender_means = (
        df.groupby("gender")[list(available_map.values())]
        .mean()
        .reset_index()
    )

    gender_means.rename(columns=reverse_map, inplace=True)

    fig5 = px.bar(
        gender_means,
        x="gender",
        y=list(reverse_map.values()),
        barmode="group",
        title="Emotional Resilience Attributes by Gender"
    )

    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Gender variable is not available for group comparison.")
