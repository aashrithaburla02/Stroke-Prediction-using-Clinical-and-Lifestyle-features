import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Stroke Prediction", page_icon="🧠")

st.title("🧠 Stroke Prediction Web App")
st.write("Fill the details below and click Predict.")

# ---- Reset Button ----
if st.button("Reset Form"):
    st.session_state.clear()
    st.rerun()

# ---- User Inputs ----
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=0)

hypertension = st.selectbox(
    "Hypertension",
    options=[0, 1],
    format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)"
)

heart_disease = st.selectbox(
    "Heart Disease",
    options=[0, 1],
    format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)"
)

ever_married = st.selectbox("Ever Married", ["Yes", "No"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

residence = st.selectbox("Residence Type", ["Urban", "Rural"])

avg_glucose = st.number_input("Average Glucose Level", value=0.0)

bmi = st.number_input("BMI", value=0.0)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# ---- Prediction ----
if st.button("Predict Stroke Risk"):

    input_dict = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[model_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠ High Risk of Stroke (Yes)")
    else:
        st.success("✅ Low Risk of Stroke (No)")

    st.write(f"Stroke Probability: {probability * 100:.2f}%")

