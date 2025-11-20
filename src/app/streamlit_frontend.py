import streamlit as st
import json
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model locally
@st.cache_resource
def load_model():
    model = joblib.load("models/student_dropout_model.pkl")
    return model

model = load_model()

# Local prediction instead of API call
def predict_dropout(data):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]   # dropout probability
    return {"prediction": float(proba)}

# Title and description
st.title("🎓 Student Dropout Risk Predictor")
st.markdown("""
This app predicts the probability of a student dropping out based on academic and demographic factors.
""")

# Sidebar input
st.sidebar.header("Input Student Data")

age = st.sidebar.number_input("Age at enrollment", 16, 70, 19)
cu_1st_approved = st.sidebar.number_input("Curricular units 1st sem approved", 0, 20, 6)
cu_2nd_approved = st.sidebar.number_input("Curricular units 2nd sem approved", 0, 20, 6)
cu_1st_no_eval = st.sidebar.number_input("Curricular units 1st sem without evaluations", 0, 20, 0)
cu_2nd_no_eval = st.sidebar.number_input("Curricular units 2nd sem without evaluations", 0, 20, 0)
cu_1st_grade = st.sidebar.number_input("Curricular units 1st sem grade", 0.0, 20.0, 12.0)
cu_2nd_grade = st.sidebar.number_input("Curricular units 2nd sem grade", 0.0, 20.0, 12.0)
tuition_fees = st.sidebar.selectbox("Tuition fees up to date", [0, 1])
scholarship = st.sidebar.selectbox("Scholarship holder", [0, 1])

if st.sidebar.button("Predict Dropout Risk"):
    input_data = {
        "Age_at_enrollment": age,
        "Curricular_units_1st_sem_approved": cu_1st_approved,
        "Curricular_units_2nd_sem_approved": cu_2nd_approved,
        "Curricular_units_1st_sem_without_evaluations": cu_1st_no_eval,
        "Curricular_units_2nd_sem_without_evaluations": cu_2nd_no_eval,
        "Curricular_units_1st_sem_grade": cu_1st_grade,
        "Curricular_units_2nd_sem_grade": cu_2nd_grade,
        "Tuition_fees_up_to_date": tuition_fees,
        "Scholarship_holder": scholarship
    }

    result = predict_dropout(input_data)

    if "prediction" in result:
        percentage = round(result["prediction"] * 100, 2)

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            # Donut chart
            fig = px.pie(
                names=["Dropout Risk", "Safe"],
                values=[percentage, 100 - percentage],
                hole=0.5,
                color_discrete_sequence=["#ff6b6b", "#4ecdc4"]
            )
            fig.update_traces(textinfo="none")
            fig.update_layout(
                title="Dropout Risk Probability",
                showlegend=True,
                width=350,
                height=350
            )
            st.plotly_chart(fig)

        with col2:
            st.subheader("📊 Prediction Details")
            st.metric(label="Dropout Probability", value=f"{percentage}%")

            # Risk category
            if percentage >= 60:
                risk = "High Risk"
            elif percentage >= 30:
                risk = "Moderate Risk"
            else:
                risk = "Low Risk"

            # Risk indicator box
            color = "#ff6b6b" if risk == "High Risk" else "#feca57" if risk == "Moderate Risk" else "#4ecdc4"
            st.markdown(f"""
            <div style="padding:10px;background-color:{color};border-radius:5px;">
                <h1 style="color:white;text-align:center;">{risk}</h1>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Prediction Error occurred.")

# Explanatory notes
with st.expander("ℹ️ How to use this app"):
    st.write("""
    - Enter student details in the sidebar.
    - Click **Predict Dropout Risk** to generate probability.
    - The donut chart visualizes dropout percentage.
    - Color-coded risk indicator shows severity.
    """)

