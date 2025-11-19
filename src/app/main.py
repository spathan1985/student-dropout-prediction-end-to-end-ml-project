"""
End-to-End Student Dropout Prediction Project
Author: Shagufta Pathan
Dataset: Predict Students’ Dropout and Academic Success (UCI/MDPI)
"""

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
# import asyncio
# import nest_asyncio
import joblib 
import os 


# ===============================
# 2. FastAPI App Initialization
# ===============================
app = FastAPI(title="Student Dropout Predictor API")


# ===============================
# 3. Pydantic Model for Input
# ===============================
class StudentData(BaseModel):
    Age_at_enrollment: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_without_evaluations: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_2nd_sem_grade: float
    Tuition_fees_up_to_date: int
    Scholarship_holder: int

# ===============================
# 4. Load Model at Startup
# ===============================
# Load pretrained model once at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "./models/student_dropout_model.pkl"   # adjust path if needed
    # model_path = os.getenv("MODEL_PATH", "/code/models/student_dropout_model.pkl")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

# ===============================
# 5. Root Endpoint
# ===============================
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Student Dropout Predictor API",
        "version": "1.0",
        "docs_url": "/docs",
        "endpoints": {
            "predict": "/predict",
            "documentation": "/docs",
            "openapi": "/openapi.json"
        }
    }

# ===============================
# 6. Predict Endpoint
# ===============================
@app.post("/predict")
def predict(data: StudentData):
    global model
    if model is None:
        return {"error": "Model not loaded. Ensure 'student_dropout_model.pkl' exists."}

    X = pd.DataFrame([data.dict()])
    try:
        pred_proba = model.predict_proba(X)[0][1]
        risk = "High Risk" if pred_proba > 0.6 else "Low Risk"
        return {
            "dropout_probability": round(float(pred_proba), 3),
            "risk_category": risk
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


# ===============================
# 7. Run the API
# ===============================
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
