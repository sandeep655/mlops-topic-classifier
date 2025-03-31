import os
import streamlit as st
import requests

# Require API_URL from environment
API_URL = os.getenv("API_URL")
if not API_URL:
    st.error("API_URL environment variable is not set. Please configure it in Streamlit secrets or your environment.")
    st.stop()

st.title("Topic Classifier: Computer Hardware vs Baseball")

text = st.text_area("Enter a sentence below for pred:")
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