import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Dropout Risk Predictor")
st.markdown("""
This app predicts the probability of a student dropping out based on academic and demographic factors.
""")

def predict_dropout(data):
    # Update URL based on deployment (local/AWS)
    API_URL = "http://3.139.61.120:80/predict" #  AWS URL
    
    try:
        response = requests.post(API_URL, json=data)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# Create form
with st.form("prediction_form"):
    st.subheader("Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age at Enrollment", 
                            min_value=15, 
                            max_value=70, 
                            value=20)
        
        sem1_approved = st.number_input("1st Semester Approved Units",
                                      min_value=0,
                                      max_value=10,
                                      value=5)
        
        sem2_approved = st.number_input("2nd Semester Approved Units",
                                      min_value=0,
                                      max_value=10,
                                      value=5)
        
        sem1_missed = st.number_input("1st Semester Units Without Evaluation",
                                    min_value=0,
                                    max_value=10,
                                    value=0)
        
    with col2:
        sem2_missed = st.number_input("2nd Semester Units Without Evaluation",
                                    min_value=0,
                                    max_value=10,
                                    value=0)
        
        sem1_grade = st.number_input("1st Semester Average Grade",
                                   min_value=0.0,
                                   max_value=20.0,
                                   value=12.0)
        
        sem2_grade = st.number_input("2nd Semester Average Grade",
                                   min_value=0.0,
                                   max_value=20.0,
                                   value=12.0)
        
        fees_uptodate = st.selectbox("Tuition Fees Up to Date?",
                                   options=["Yes", "No"],
                                   index=0)
        
        scholarship = st.selectbox("Scholarship Holder?",
                                options=["Yes", "No"],
                                index=0)

    submit_button = st.form_submit_button("Predict Dropout Risk")

    if submit_button:
        # Prepare data
        input_data = {
            "Age_at_enrollment": age,
            "Curricular_units_1st_sem_approved": sem1_approved,
            "Curricular_units_2nd_sem_approved": sem2_approved,
            "Curricular_units_1st_sem_without_evaluations": sem1_missed,
            "Curricular_units_2nd_sem_without_evaluations": sem2_missed,
            "Curricular_units_1st_sem_grade": sem1_grade,
            "Curricular_units_2nd_sem_grade": sem2_grade,
            "Tuition_fees_up_to_date": 1 if fees_uptodate == "Yes" else 0,
            "Scholarship_holder": 1 if scholarship == "Yes" else 0
        }
        
        # Get prediction
        result = predict_dropout(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        if "error" not in result:
            prob = result["dropout_probability"]
            risk = result["risk_category"]
            
            # Probability gauge
            with col1:
                fig = px.pie(values=[prob, 1-prob], 
                           names=['Dropout Risk', 'Retention Probability'],
                           hole=0.7,
                           color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
                fig.update_layout(title="Dropout Probability")
                st.plotly_chart(fig)
            
            # Risk category
            with col2:
                st.markdown(f"""
                ### Risk Assessment
                - **Category:** {risk}
                - **Probability:** {prob:.1%}
                """)
                
                # Color-coded risk indicator
                color = "#ff6b6b" if risk == "High Risk" else "#4ecdc4"
                st.markdown(f"""
                <div style="padding:10px;background-color:{color};border-radius:5px;">
                    <h1 style="color:white;text-align:center;">{risk}</h1>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error(f"Prediction Error: {result['error']}")

# Add explanatory notes
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. Enter the student's academic information in the form above
    2. Click 'Predict Dropout Risk' to get the prediction
    3. The model will return:
        - Dropout probability (0-100%)
        - Risk category (High/Low)
        - Visualization of the prediction
    """)