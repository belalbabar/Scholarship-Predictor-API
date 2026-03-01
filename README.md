# 🎓 Scholarship Prediction Web Application

A Machine Learning powered web application that predicts whether a student is likely to receive a scholarship based on academic profile data.

The model is deployed using **FastAPI** and hosted on Railway.

---

## 🚀 Live Demo

🔗 https://web-production-cc6cf.up.railway.app/  

📄 Interactive API Docs:  
https://web-production-cc6cf.up.railway.app/docs

---

## 🧠 Features

- Predicts scholarship eligibility (0 or 1)
- Returns prediction probability
- Built using Random Forest Classifier
- Data preprocessing with StandardScaler
- FastAPI backend deployment
- CORS enabled for frontend integration
- Live deployment on Railway

---

## 📊 Input Parameters

| Parameter | Type | Description |
|-----------|------|------------|
| GRE_Score | int | GRE Score |
| TOEFL_Score | int | TOEFL Score |
| University_Rating | int | University Rating (1–5) |
| Statement_of_Purpose | float | SOP strength (0–5) |
| Letter_of_recommendation | float | LOR strength (0–5) |
| CGPA | float | Undergraduate CGPA |
| Research | int | Research Experience (0 or 1) |

---

## 📦 Tech Stack

- Python
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Railway

---
