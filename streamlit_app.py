import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("Topic Classification - Computer Hardware vs Baseball")
user_input = st.text_area("Enter a sentence['there is issue in my RAM'] AND please enable score:")
show_score = st.checkbox("Show confidence score")

# Add a button to trigger predictions
if st.button("Predict"):
    text_vector = vectorizer.transform([user_input])
    prediction = model.predict(text_vector)[0]
    proba = model.predict_proba(text_vector)[0]
    label = "Baseball-related" if prediction == 1 else "Computer Hardware-related"
    st.write(f"Prediction: {label}")
    if show_score:
        st.write(f"Confidence: {np.max(proba) * 100:.2f}%")
