import gradio as gr
import requests
import json

def predict(age, sem1_approved, sem2_approved, sem1_missed, sem2_missed, 
           sem1_grade, sem2_grade, fees_uptodate, scholarship):
    
    # Prepare input data
    input_data = {
        "Age_at_enrollment": age,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Curricular_units_1st_sem_without_evaluations": sem1_missed,
        "Curricular_units_2nd_sem_without_evaluations": sem2_missed,
        "Curricular_units_1st_sem_grade": sem1_grade,
        "Curricular_units_2nd_sem_grade": sem2_grade,
        "Tuition_fees_up_to_date": fees_uptodate,
        "Scholarship_holder": scholarship
    }
    
    # Make API request
    API_URL = "http://18.216.149.175:80/predict"  #  AWS URL
    response = requests.post(API_URL, json=input_data)
    result = json.loads(response.text)
    
    return f"Dropout Probability: {result['dropout_probability']:.1%}\nRisk Category: {result['risk_category']}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age at Enrollment"),
        gr.Number(label="1st Semester Approved Units"),
        gr.Number(label="2nd Semester Approved Units"),
        gr.Number(label="1st Semester Units Without Evaluation"),
        gr.Number(label="2nd Semester Units Without Evaluation"),
        gr.Number(label="1st Semester Average Grade"),
        gr.Number(label="2nd Semester Average Grade"),
        gr.Checkbox(label="Tuition Fees Up to Date"),
        gr.Checkbox(label="Scholarship Holder")
    ],
    outputs="text",
    title="Student Dropout Risk Predictor",
    description="Enter student information to predict dropout risk"
)

iface.launch(server_name="0.0.0.0", server_port=7860)