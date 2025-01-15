import streamlit as st
from joblib import load
import numpy as np

model = load('model.joblib')

st.title("COVID-19 Prediction App By vikas")

st.header("Enter Patient Details:")

age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

gender = st.selectbox("Gender", ["Male", "Female"])

fever = st.number_input("Fever (in °F)", min_value=95.0, max_value=110.0, value=98.6, step=0.1)

cough = st.selectbox("Cough Severity", ["Mild", "Strong"])

city = st.selectbox("City", ["Delhi", "Kolkata", "Mumbai", "Bangalore"])

def preprocess_input(age, gender, fever, cough, city):
    gender_encoded = 1 if gender == "Male" else 0
    cough_encoded = 1 if cough == "Mild" else 2
    city_encoded = {"Delhi": 1, "Kolkata": 2, "Mumbai": 3, "Bangalore": 4}[city]
    
    return np.array([[age, gender_encoded, fever, cough_encoded, city_encoded]])

if st.button("Predict"):
    input_data = preprocess_input(age, gender, fever, cough, city)

    prediction = model.predict(input_data)

    result = "Yes" if prediction[0] == 1 else "No"
    st.subheader(f"Prediction: {result}")
