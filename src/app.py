import streamlit as st
import pandas as pd
import joblib

# Load trained model, scaler, and feature columns
model = joblib.load('svm_sleep_disorder_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("😴 Sleep Disorder Prediction")
st.write("Enter your details below to see if you might be at risk for a sleep disorder based on our trained model.")

# --- INPUT FORM ---
with st.form("user_input_form"):
    screen_time = st.number_input("Screen Time (hours per day)", min_value=0.0, max_value=24.0, value=5.0)
    sleep_duration = st.number_input("Sleep Duration (hours per night)", min_value=0.0, max_value=24.0, value=8.0)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 3)
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=130, value=80)
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal Weight", "Overweight", "Obese"])

    submitted = st.form_submit_button("Check for Sleep Disorder")

# --- PREDICTION ---
if submitted:
    input_dict = {
        'Screen Time': screen_time,
        'Sleep Duration': sleep_duration,
        'Stress Level': stress_level,
        'Systolic': systolic,
        'Diastolic': diastolic,
        'Age': age,
        'Gender': gender,
        'BMI Category': bmi_category,
        # simulate original feature engineering
        'Alcohol/Smoking Addiction': 0,  # default unless detected
        'Sleep Walking': 0  # default unless detected
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical variables like original training
    input_df = pd.get_dummies(input_df)

    # Ensure all required columns are present
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing column with 0

    # Reorder columns
    input_df = input_df[model_columns]

    # Scale numeric features (only those that were scaled during training)
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    st.subheader("🧠 Prediction Result:")
    st.success(f"Based on your input, the predicted sleep disorder is: **{prediction}**.")
