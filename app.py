import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🎯 Customer Segmentation App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("marketing_campaign1.xlsx")

df = load_data()

# Select features for clustering
features = df[["Income", "Age", "Spending Score"]].dropna()

# Train KMeans model
@st.cache_resource
def train_model():
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(features)
    return model

model = train_model()

# Display data preview
st.subheader("📊 Preview of Data")
st.dataframe(df.head())

# Input features for prediction
st.sidebar.header("🔍 Input Customer Details")
try:
    income = st.sidebar.slider("Income", min_value=0, max_value=200000, value=50000, step=1000)
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35)
    score = st.sidebar.slider("Spending Score", min_value=1, max_value=100, value=50)

    input_data = np.array([[income, age, score]])

    if st.sidebar.button("Predict Segment"):
        prediction = model.predict(input_data)
        st.success(f"🎯 Predicted Segment: {int(prediction[0])}")
except Exception as e:
    st.error(f"Error in prediction: {e}")

