import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# -------------------------
# Path setup (สำคัญมาก)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# -------------------------
# Load model
# -------------------------
model = joblib.load(os.path.join(MODEL_DIR, "spam_lr_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Spam Detection API")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(data: TextInput):
    X = vectorizer.transform([data.text])
    proba = model.predict_proba(X)[0]

    return {
        "ham_probability": float(proba[0]),
        "spam_probability": float(proba[1]),
        "prediction": "spam" if proba[1] > 0.4 else "ham"
    }

@app.get("/", response_class=HTMLResponse)
def read_root():
    file_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()