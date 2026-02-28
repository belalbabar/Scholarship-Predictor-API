import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI()

# Base directory for loading pickles
BASE_DIR = os.path.dirname(__file__)

# Load model and scaler safely
try:
    rf_model = joblib.load(os.path.join(BASE_DIR, "scholarship_model.pkl"))
except FileNotFoundError:
    raise FileNotFoundError("scholarship_model.pkl not found in the project folder")

try:
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
except FileNotFoundError:
    raise FileNotFoundError("scaler.pkl not found in the project folder")


# Input model
class StudentData(BaseModel):
    GRE_Score: int
    TOEFL_Score: int
    University_Rating: int
    SOP: float
    LOR: float
    CGPA: float
    Research: int


@app.post("/predict_scholarship")
def predict(data: StudentData):
    try:
        # Prepare input for model
        input_data = np.array([[data.GRE_Score, data.TOEFL_Score, data.University_Rating,
                                data.SOP, data.LOR, data.CGPA, data.Research]])
        input_scaled = scaler.transform(input_data)
        prediction = rf_model.predict(input_scaled)[0]
        probability = rf_model.predict_proba(input_scaled)[0][1]

        return {"Scholarship": int(prediction), "Probability": round(probability, 2)}

    except Exception as e:
        # Return any error as JSON
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/")
def home():
    return {"message": "Scholarship Prediction API is running"}