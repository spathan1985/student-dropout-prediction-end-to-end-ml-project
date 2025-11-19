# 🎓 Student Dropout Risk Prediction

An end-to-end machine learning project to **predict which students are at risk of dropping out** of high school or college — built using **Pandas, LightGBM, MLflow, FastAPI, Docker, and AWS Lambda**.

---

## 📘 Project Overview

Educational institutions face increasing pressure to identify students at risk of dropping out early.  
This project builds a data-driven solution that predicts dropout risk based on academic, socio-economic, and demographic data.

### 🎯 Objectives
- Predict individual student dropout risk (probability and category).
- Provide interpretable feature insights for educators.
- Deploy a lightweight, serverless prediction API to AWS Lambda.

---

## 🧱 Tech Stack

| Layer | Tools |
|:------|:------|
| **Data** | Python, Pandas, NumPy |
| **Modeling** | LightGBM, Scikit-learn |
| **Tracking** | MLflow |
| **API** | FastAPI |
| **Containerization** | Docker |
| **Deployment** | AWS ECS |
| **Optional** | Streamlit (dashboard), Optuna (tuning), SHAP (interpretability) |

---

## 🗂️ Project Structure

```
student_dropout_predictor/
├── data/
│   └── raw/student_dropout_data.csv
├── src/
│   └── train_and_save_model.py
├── models/
│   └── student_dropout_model.pkl
├── notebooks/
│   └── student_dropout_eda.ipynb
├── requirements.txt
└── README.md
└── Dockerfile
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Predicting-Student-Dropout-Risk.git
cd student-dropout-predictor
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Data
Data Source: [Student Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

Store CSV in `data/raw/`.

### 5. Data Preparation and Model Training
```bash
python src/train_and_save_model.py
```
Model artifacts will be saved under `models/`.

### 6. Run FastAPI Locally
```bash
uvicorn src.app.main:app --reload
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) to test the API.

---

## 🧠 Example API Usage

**POST /predict**

Request:
```json
{
  "Age_at_enrollment": 40,
  "Curricular_units_1st_sem_approved": 2,
  "Curricular_units_2nd_sem_approved": 5,
  "Curricular_units_1st_sem_without_evaluations": 1,
  "Curricular_units_2nd_sem_without_evaluations": 2,
  "Curricular_units_1st_sem_grade": 14.0,
  "Curricular_units_2nd_sem_grade": 14.0,
  "Tuition_fees_up_to_date": 0,
  "Scholarship_holder": 0
}
```

Response:
```json
{
  "dropout_probability":0.885,
  "risk_category":"High Risk"
}
```

---

## 🧾 MLflow Tracking

Start MLflow locally to visualize experiments:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) to view logs of parameters, metrics, and artifacts.

---

## 🐳 Docker Containerization

1. Build and run Docker image:
```bash
docker build -t student-dropout-image-test . 
```

2. Run container:
```bash 
docker run -d --name student-dropout-container-test -p 80:80 student-dropout-image-test
```
3. Test Docker container:

Visit (http://0.0.0.0:80)⁠. It should give you the same result as before.

4. Push to Docker Hub (login first):
```bash
docker login
docker tag student-dropout-image-test spathands/student-dropout-predictor-project:latest
docker push spathands/student-dropout-predictor-project:latest
```

---

## ☁️ AWS Elastic Container Service Deployment

1. Login to AWS account and search for ECS.
2. Create a new Task Definition and give it a name:
   1. with Launch type as AWS Fargate
   2. Select Operating system/ Architecture as Linux/ARM64 (for MAC)
   3. Select Memory to 2GB
   4. Leave Task Role as None if not already created
   5. Specify Container details
3. Go to Clusters and create a new cluster
4. Create Service and (optionally) Application Load Balancer to expose the API.
5. Ensure security groups / IAM roles allow traffic and that environment variable MODEL_PATH (if used) is set in the container/task definition or model is baked into image.
6. Go to Tasks within the Service name we created, and use the Public IP to test the deployment.

---

## Streamlit App Test
```bash
streamlit run src/app/streamlit_frontend.py --server.port 8501 --server.address 0.0.0.0
```

## Containerize Streamlit frontend (Optional)
1. Build and run:
```bash
docker build -t student-dropout-frontend -f Dockerfile.frontend .
docker run -d -p 8501:8501 --name student-dropout-predictor-ui student-dropout-frontend
```

2. Push frontend image to Docker Hub
```bash
docker tag student-dropout-frontend spathands/student-dropout-predictor-project:ui
docker push spathands/student-dropout-predictor-project:ui
```


## 📊 Model Evaluation Metrics

| Metric | Description |
|:--------|:-------------|
| Accuracy | Overall correctness |
| Precision / Recall / F1 | Class-level performance |
| AUC-ROC | Ability to rank at-risk students correctly |
| SHAP | Feature interpretability |

---

## 🗓️ Roadmap

- Data ingestion and cleaning  
- Baseline model  
- MLflow tracking  
- FastAPI service  
- Docker containerization  
- AWS ECS deployment  
- Streamlit dashboard  

---

## 🔍 Insights from EDA
*(from notebooks/student_dropout_eda.ipynb)*


📊 Insights from Exploratory Data Analysis

**1. Imbalanced Outcomes:**
- The dataset shows a strong class imbalance — far fewer students drop out compared to those who graduate or remain enrolled.

- This means class weighting or resampling (SMOTE) will be needed during modeling.

**2. Early Academic Performance Drives Outcomes:**

- Curricular units approved and failed in the 1st semester are the strongest predictors of dropout risk.

- Students with early failures or lower first-semester grades are more likely to drop out.

**3. Grades Correlate Highly Across Semesters:**

- Strong positive correlation (≈ 0.75–0.85) between 1st- and 2nd-semester grades and credits suggests consistency in performance — early success predicts later success.

**4. Socio-Economic Factors Matter:**

- Students who receive scholarships or stay current on tuition fees are much less likely to drop out.

- Financial stress indicators align closely with dropout patterns.

**5. Age at Enrollment Shows a U-Shaped Pattern:**

- Both younger (< 20) and older (> 30) students exhibit higher dropout rates.

- Mid-range ages (21–27) are the most stable cohort.

**6. Gender Differences Are Modest but Noticeable:**

- Male students show slightly higher dropout tendencies, especially when combined with poor academic performance or delayed tuition.

**7. Displacement & Special Needs Status:**

- Displaced students (e.g., those who moved regions) have higher dropout risk — possibly due to adjustment or resource challenges.

**8. Strong Feature Correlations Indicate Redundancy:**

- Academic performance metrics (approved, failed, grades) are inter-correlated; dimensionality reduction or feature selection will improve model efficiency.

**9. Target Leakage Check Passed:**

- No direct data leakage was found — features are available early enough in the academic cycle to be used for early-warning prediction.

**10. SHAP & Interpretability Ready:**

- The relationships are interpretable and ideal for SHAP analysis later, making this dataset perfect for both predictive and explainable AI storytelling.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to modify.

---

## 🧑‍💻 Author

**Shagufta Pathan**  
*Data Scientist | Supply Chain & AI Enthusiast*  
[LinkedIn](https://linkedin.com/in/your-profile) • [Medium](https://medium.com/@your-handle)

---