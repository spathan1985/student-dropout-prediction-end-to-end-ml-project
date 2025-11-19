"""
Train and Save Student Dropout Model
Author: Shagufta Pathan
"""

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import joblib 
import os 

# Add after imports
SELECTED_FEATURES = [
    'Age_at_enrollment',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_without_evaluations',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade',
    'Tuition_fees_up_to_date',
    'Scholarship_holder'
]

MODEL_OUTPUT_PATH = "/Users/shagufta/Documents/Projects/Student Dropout Risk Prediction/models/student_dropout_model.pkl"

# ===============================
# 2. Load & Clean Data
# ===============================
def load_data(path: str):
    df = pd.read_csv(path, sep=';') if path.endswith('.csv') else pd.read_excel(path)
    print(f"Loaded dataset with shape: {df.shape}")
    # print(f"Loaded dataset with shape: {df.head}")
    return df

def clean_data(df):
    # Drop duplicates & irrelevant columns
    df = df.drop_duplicates()
    df = df.dropna(subset=['Status'])
    # Encode Status
    df['Status'] = df['Status'].replace({'Dropout': 1, 'Enrolled': 0, 'Graduate': 0})
    return df

# ===============================
# 3. Feature Engineering
# ===============================
def feature_engineering(df):
    # Simple numeric feature cleanup
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Encode categoricals
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df

# ===============================
# 4. Train Model + Log to MLflow
# ===============================
def train_model(df):
    
    X = df[SELECTED_FEATURES]
    print("cols used to train model", X.columns)
    # X = df.drop(columns=['Status'])
    y = df['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )

    mlflow.set_experiment("student_dropout_risk")
    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probas)

        print("Accuracy:", acc)
        print("ROC-AUC:", auc)
        print(classification_report(y_test, preds))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.lightgbm.log_model(model, "model")

         # Save model
        # os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        mlflow.lightgbm.log_model(model, "model")

    return model, X_test, y_test
    # # SHAP feature importance
    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_test)
    # shap.summary_plot(shap_values, X_test, show=False)

    # import matplotlib.pyplot as plt

def plot_shap_summary(model, X_test):
    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Create detailed plot with feature impacts
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_detailed.png')
    plt.close()



# ===============================
# Run the Full Pipeline
# ===============================
if __name__ == "__main__":

    data_path = "/Users/shagufta/Documents/Projects/Student Dropout Risk Prediction/data/raw/student_dropout_data.csv"
    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    # print(df.columns)
    model, X_test, y_test = train_model(df)
    
    # Add this line to generate SHAP plots
    plot_shap_summary(model, X_test)
    
    print("Model trained and ready to deploy!")