import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Heathcare Dashboard",page_icon="🤖" ,layout="wide")
st.title("🤖 Healthcare Dashboard")
st.markdown("Enter the feature below to predict the outcome.")
st.sidebar.header("About")
st.sidebar.info("This app uses a trained ML model to make predictions.")
@st.cache_resource
def load_model():
    import pickle
    with open('model_new.pkl','rb')as file:
        model=pickle.load(file)
    return model
model=load_model()

st.header("Input Features")
col1,col2=st.columns(2)
with col1:
    age=st.selectbox("Age Group",["Young","Middle-Aged","Senior"])
    gender=st.selectbox("Gender",["Male","Female"])
    blood_type=st.selectbox("Blood Type",["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    medical_conditions=st.selectbox("Medical Conditions",["Diabetes","Cancer","Asthma","Arthritis","Hypertension","Obesity","None"])
with col2:
    insurance_provider=st.selectbox("Insurance Provider",["Cigna","Blue Cross","UnitedHealthcare","Medicare","Aetna"])
    admission_type=st.selectbox("Admission Type",["Emergency","Urgent","Elective"])
    medication=st.selectbox("Medication",["paracetamol","Ibuprofen","Aspirin","Penicillin","Lipitor","None"])

if st.button("Predict Outcome"):
    st.spinner("Predicting...")
    input_data=pd.DataFrame([
        {
            'Age_Group':age,
            'Gender':gender,
            'Blood Type':blood_type,
            'Medical Condition':medical_conditions,
            'Insurance Provider':insurance_provider,
            'Admission Type':admission_type,
            'Medication':medication
        }
    ])
    prediction=model.predict(input_data)
    st.markdown("<h3 style='color:#4CAF50;'>Prediction Result</h3>", unsafe_allow_html=True)
    st.success(f"The predicted outcome is: {prediction[0]}")
