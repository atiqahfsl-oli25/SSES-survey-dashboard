import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("🧠 Emotional Resilience Analysis")

# Load specific cleaned data for this analysis
DATA_URL = "https://raw.githubusercontent.com/nhusna01/SSES-survey-dashboard/main/Hafizah_SSES_Cleaned.csv"
df = pd.read_csv(DATA_URL)

objective3_cols = ['calm_under_pressure', 'emotional_control', 'adaptability', 'self_motivation', 'task_persistence', 'teamwork']
df[objective3_cols] = df[objective3_cols].apply(pd.to_numeric, errors="coerce").fillna(df[objective3_cols].median())

# Metrics
agree_prop = df[objective3_cols].isin([4, 5]).mean()
m1, m2, m3 = st.columns(3)
m1.metric("Avg. Resilience", f"{agree_prop.mean():.1%}")
m2.metric("Strongest", agree_prop.idxmax().replace('_', ' ').title())
m3.metric("Growth Area", agree_prop.idxmin().replace('_', ' ').title())

# Radar Chart
st.subheader("Resilience Profile")
mean_scores = df[objective3_cols].mean()
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=mean_scores.values.tolist() + [mean_scores.values[0]],
    theta=[c.replace('_', ' ').title() for c in objective3_cols] + [objective3_cols[0].replace('_', ' ').title()],
    fill='toself'
))
st.plotly_chart(fig_radar, use_container_width=True)

# Heatmap
st.subheader("Correlation Matrix")
fig_corr = px.imshow(df[objective3_cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
st.plotly_chart(fig_corr, use_container_width=True)
