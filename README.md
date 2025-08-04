# ritik-health-test-classifier
Streamlit app that predicts medical test results (Normal, Abnormal, or Inconclusive) based on patient info using a logistic regression model.
# 🩺 Medical Test Result Classifier

This is a simple yet powerful **Streamlit web app** that predicts whether a patient's medical test result is **Normal**, **Abnormal**, or **Inconclusive** using a **Logistic Regression model** trained on synthetic medical data.

## 🚀 Live Demo
Try the app here: # 🩺 Medical Test Result Classifier

This is a simple yet powerful **Streamlit web app** that predicts whether a patient's medical test result is **Normal**, **Abnormal**, or **Inconclusive** using a **Logistic Regression model** trained on synthetic medical data.

## 🚀 Live Demo
Try the app here: https://ritik-health-test-classifier-ggzevwtpvciw6vsjyzugca.streamlit.app/


---

## 🧠 How It Works

You enter:
- Age  
- Gender  
- Blood Type  
- Medical Condition  
- Admission Type  
- Medication  
- Day of the week

👉 Based on this data, the app predicts the likely medical test outcome using a trained logistic regression model (`.pkl` file).

## 📂 Project Structure
├── app.py # Main Streamlit app
├── logistic_regression_model_health.pkl # Trained logistic regression model
├── requirements.txt # Python dependencies
├── README.md # Project documentation
