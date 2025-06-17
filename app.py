import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🎯 Customer Segmentation App")

# Load model
@st.cache_resource
def load_model():
    with open("C:/Users/viral/Desktop/CSFILESProject/segmentation_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("C:/Users/viral/Desktop/CSFILESProject/marketing_campaign1.xlsx")

df = load_data()

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
