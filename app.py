import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


with open('logistic_regression_model_health.pkl', 'rb') as file:

    model = pickle.load(file)


st.set_page_config(page_title="🩺 Health Test Classifier", page_icon="💉", layout="centered")


st.markdown("<h1 style='text-align: center; color: #3B82F6;'>🩺 Medical Test Result Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a patient's medical test is <strong>Normal</strong>, <strong>Abnormal</strong>, or <strong>Inconclusive</strong>.</p>", unsafe_allow_html=True)
st.markdown("---")


with st.form("patient_form"):
    st.subheader("📋 Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👤 Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("🚻 Gender", ['Male', 'Female'])
        day = st.selectbox("📅 Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    with col2:
        blood_type = st.selectbox("🩸 Blood Type", ['B-', 'A+', 'A-', 'O+', 'AB+', 'AB-', 'B+', 'O-'])
        medical_condition = st.selectbox("🏥 Medical Condition", ['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis'])
        admission_type = st.selectbox("🚑 Admission Type", ['Urgent', 'Emergency', 'Elective'])

    medication = st.selectbox("💊 Medication", ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin', 'Lipitor'])

    submit_btn = st.form_submit_button("🔍 Predict")


if submit_btn:

    days_mapping = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3,
        "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
    }
    days_numeric = days_mapping[day]


    input_data = pd.DataFrame({
        'Age': [age],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Blood Type_A-': [1 if blood_type == 'A-' else 0],
        'Blood Type_AB+': [1 if blood_type == 'AB+' else 0],
        'Blood Type_AB-': [1 if blood_type == 'AB-' else 0],
        'Blood Type_B+': [1 if blood_type == 'B+' else 0],
        'Blood Type_B-': [1 if blood_type == 'B-' else 0],
        'Blood Type_O+': [1 if blood_type == 'O+' else 0],
        'Blood Type_O-': [1 if blood_type == 'O-' else 0],
        'Medical Condition_Asthma': [1 if medical_condition == 'Asthma' else 0],
        'Medical Condition_Cancer': [1 if medical_condition == 'Cancer' else 0],
        'Medical Condition_Diabetes': [1 if medical_condition == 'Diabetes' else 0],
        'Medical Condition_Hypertension': [1 if medical_condition == 'Hypertension' else 0],
        'Medical Condition_Obesity': [1 if medical_condition == 'Obesity' else 0],
        'Admission Type_Emergency': [1 if admission_type == 'Emergency' else 0],
        'Admission Type_Urgent': [1 if admission_type == 'Urgent' else 0],
        'Medication_Ibuprofen': [1 if medication == 'Ibuprofen' else 0],
        'Medication_Lipitor': [1 if medication == 'Lipitor' else 0],
        'Medication_Paracetamol': [1 if medication == 'Paracetamol' else 0],
        'Medication_Penicillin': [1 if medication == 'Penicillin' else 0],
        'Days_numeric': [days_numeric]
    })

   
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

 
    prediction = model.predict(input_data)
    result_mapping = {0: '🟢 Normal', 1: '🔴 Abnormal', 2: '🟡 Inconclusive'}
    result = result_mapping[prediction[0]]

    # Display result
    st.success(f"🎯 **Prediction Result:** {result}")

