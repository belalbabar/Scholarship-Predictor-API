import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -----------------------
# Initialize FastAPI app
# -----------------------
app = FastAPI()

# Allow CORS for any origin (your HTML JS can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load model and scaler
# -----------------------
rf_model = joblib.load("scholarship_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------
# Input model schema
# -----------------------
class StudentData(BaseModel):
    GRE_Score: int
    TOEFL_Score: int
    University_Rating: int
    Statement_of_Purpose: float
    Letter_of_recommendation: float
    CGPA: float
    Research: int

# -----------------------
# API endpoint
# -----------------------
@app.post("/predict_scholarship")
def predict(data: StudentData):
    input_data = np.array([[ 
        data.GRE_Score,
        data.TOEFL_Score,
        data.University_Rating,
        data.Statement_of_Purpose,
        data.Letter_of_recommendation,
        data.CGPA,
        data.Research
    ]])
    
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]

    return {
        "Scholarship": int(prediction),
        "Probability": round(probability, 2)
    }

# -----------------------
# Serve HTML file
# -----------------------
# Mount current folder as static files
# index.html will be served at root /
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# -----------------------
# Optional root API test
# -----------------------
@app.get("/api")
def home():
    return {"message": "Scholarship Prediction API is running"}