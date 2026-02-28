import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

rf_model = joblib.load("scholarship_model.pkl")
scaler = joblib.load("scaler.pkl")
app = FastAPI()

class StudentData(BaseModel):
    GRE_Score : int
    TOEFL_Score : int
    University_Rating : int
    SOP : float
    LOR : float
    CGPA : float
    Research : int

@app.post("/predict_scholarship")
def predict(data:StudentData):
    input_data = np.array([[data.GRE_Score,data.TOEFL_Score,data.University_Rating,data.SOP,data.LOR,
                            data.CGPA,data.Research]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]
    return {"Scholarship": int(prediction), "Probability": round(probability, 2)}
@app.get("/")
def home():
    return{'message':"Scholarship Prediction API is running"}
