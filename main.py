import streamlit as st
from preprocess import load_data

st.set_page_config(page_title="SSES Survey Dashboard", layout="wide")

st.title("SSes Survey Dashboard")
st.write("Monitoring survey responses in real time.")

# Load data
df = load_data()

# Show raw data
st.subheader("Raw Responses")
st.dataframe(df, use_container_width=True)

# Summary section
st.subheader("Summary Statistics")
st.write(df.describe(include='all'))

# Example chart — replace with your question columns
if len(df.columns) > 1:
    question_col = df.columns[1]  # choose any column
    st.subheader(f"Distribution for: {question_col}")
    st.bar_chart(df[question_col].value_counts())

