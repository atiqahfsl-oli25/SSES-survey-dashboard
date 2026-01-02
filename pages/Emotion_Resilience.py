import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================================
# PAGE HEADER
# ======================================
st.title("🧠 Emotional Resilience Analysis")

# ======================================
# PROBLEM STATEMENT & OBJECTIVE
# ======================================
with st.container():
    st.markdown("### 📝 Problem Statement")
    st.info("""
    Emotional resilience is a key factor in personal and professional success. 
    Understanding how individuals manage stress, adapt to change, control emotions, 
    and maintain motivation can help identify areas for personal development. 
    """)

    st.markdown("### 🎯 Objective")
    st.write("""
    To investigate the relationship between emotional resilience and personal development 
    attributes, including motivation, adaptability, emotional control, task persistence, 
    and teamwork skills.
    """)

# ======================================
# DATA LOADING 
# ======================================
DATA_URL = "https://raw.githubusercontent.com/nhusna01/SSES-survey-dashboard/main/dataset/Hafizah_SSES_Cleaned.csv"

@st.cache_data(ttl=3600)
def load_emotion_data():
    try:
        data = pd.read_csv(DATA_URL, storage_options={'User-Agent': 'Mozilla/5.0'})
        return data
    except Exception as e:
        st.error(f"⚠️ Could not connect to the dataset.")
        return None

df = load_emotion_data()

if df is not None:
    objective3_cols = ['calm_under_pressure', 'emotional_control', 'adaptability', 'self_motivation', 'task_persistence', 'teamwork']
    available_cols = [c for c in objective3_cols if c in df.columns]

    if not available_cols:
        st.error("❌ Required columns not found.")
    else:
        # Pre-process numeric data
        df[available_cols] = df[available_cols].apply(pd.to_numeric, errors="coerce").fillna(df[available_cols].median())

        # ======================================
        # 1. RADAR CHART
        # ======================================
        st.subheader("1. Average Resilience Profile")
        mean_scores = df[available_cols].mean()
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=mean_scores.values.tolist() + [mean_scores.values[0]],
            theta=[c.replace('_', ' ').title() for c in available_cols] + [available_cols[0].replace('_', ' ').title()],
            fill='toself', fillcolor='rgba(31, 119, 180, 0.4)', line_color='#1f77b4'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown(f"**💡 Insight:** Core group strength is **{mean_scores.idxmax().replace('_',' ').title()}**.")

        # ======================================
        # 2. CORRELATION ANALYSIS
        # ======================================
        st.subheader("2. Attribute Correlation Matrix")
        corr = df[available_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("**📌 Key Finding: Top Relationship Pairs**")
        high_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
        st.table(pd.DataFrame(high_corr[high_corr < 1].head(3), columns=['Correlation Strength']))

        # ======================================
        # 3. VIOLIN PLOT (DENSITY)
        # ======================================
        st.subheader("3. Score Density & Distribution")
        df_melted = df.melt(value_vars=available_cols, var_name="Attribute", value_name="Score")
        fig_violin = px.violin(df_melted, x="Attribute", y="Score", color="Attribute", box=True, points="all")
        st.plotly_chart(fig_violin, use_container_width=True)
        
        st.write("**Interpretation:** The width of the violin represents the frequency of scores. ")

        # ======================================
        # 4. SENTIMENT ANALYSIS (DIVERGING BAR) - FIXED SECTION
        # ======================================
        st.subheader("4. Sentiment Analysis (Agreement vs Disagreement)")
        
        def get_sentiment(series):
            # series is already the column data because of .apply()
            counts = series.value_counts(normalize=True).reindex([1,2,3,4,5], fill_value=0)
            disagree = -(counts[1] + counts[2]) * 100
            neutral = counts[3] * 100
            agree = (counts[4] + counts[5]) * 100
            return pd.Series([disagree, neutral, agree], index=['Disagree', 'Neutral', 'Agree'])

        # Apply to the dataframe and transpose
        sentiment_df = df[available_cols].apply(get_sentiment).T.reset_index()
        sentiment_df.columns = ['Attribute', 'Disagree', 'Neutral', 'Agree']

        fig_sent = px.bar(sentiment_df, x=['Disagree', 'Neutral', 'Agree'], y='Attribute', 
                          orientation='h', barmode='relative',
                          color_discrete_map={'Disagree': '#EF553B', 'Neutral': '#FECB52', 'Agree': '#00CC96'},
                          title="Diverging Likert Scale (Sentiment)")
        
        st.plotly_chart(fig_sent, use_container_width=True)
        
        st.markdown("**📌 Key Finding: Sentiment Data**")
        st.dataframe(sentiment_df)

        # ======================================
        # 5. ATTRIBUTE HIERARCHY
        # ======================================
        st.subheader("5. Attribute Hierarchy Ranking")
        tree_data = pd.DataFrame({
            "Attribute": [c.replace('_', ' ').title() for c in available_cols],
            "Mean Score": mean_scores.values
        }).sort_values(by="Mean Score", ascending=False)
        fig_tree = px.treemap(tree_data, path=['Attribute'], values='Mean Score',
                              color='Mean Score', color_continuous_scale='Blues')
        st.plotly_chart(fig_tree, use_container_width=True)

        # ======================================
        # 6. BOXPLOT (VARIABILITY)
        # ======================================
        st.subheader("6. Variability & Range Analysis")
        fig_box = px.box(df_melted, x="Attribute", y="Score", color="Attribute")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # ======================================
        # CONCLUSION
        # ======================================
        st.markdown("---")
        st.subheader("🏁 Conclusion & Recommendations")
        
        st.success(f"""
        **Final Synthesis:**
        The investigation highlights that while the group is strong in **{tree_data.iloc[0]['Attribute']}**, 
        developmental focus is needed in **{tree_data.iloc[-1]['Attribute']}**. 
        
        The correlation data suggests that emotional regulation is a "keystone" skill—improving it will likely improve all other resilience factors simultaneously.
        """)

else:
    st.warning("Please verify that the GitHub repository 'SSES-survey-dashboard' is set to **Public**.")
