from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Topic Classifier is runnings!"}

@app.post("/predict")
def predict_topic(input: InputText):
    text_vector = vectorizer.transform([input.text])
    prediction = model.predict(text_vector)[0]
    label = "Baseball-related" if prediction == 1 else "Computer Hardware-related"
    return {"prediction": label}
