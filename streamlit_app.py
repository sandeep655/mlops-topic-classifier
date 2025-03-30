import os
import streamlit as st
import requests

# Get API URL from environment variable (fallback to local)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Topic Classifier: Computer Hardware vs Baseball")

text = st.text_area("Enter a sentence for classification:")
if st.button("Classify"):
    try:
        response = requests.post(API_URL, json={"text": text})
        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.success(f"Prediction: {prediction}")
        else:
            st.error("API Error: " + response.text)
    except Exception as e:
        st.error(f"Connection error: {e}")