import streamlit as st
import pickle
import numpy as np

st.title("💧 Water Potability Prediction")
st.write("Predict whether water is potable (safe to drink) or not.")

# Load models
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Model selection
model_choice = st.selectbox("Select the model", ["Random Forest", "Logistic Regression"])

# Input fields
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
Solids = st.number_input("Solids (mg/L)", min_value=0.0, value=5000.0)
Chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, value=5.0)
Sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
Conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
Organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=80.0)
Turbidity = st.number_input("Turbidity", min_value=0.0, value=3.0)

# Prediction
if st.button("Predict Potability"):
    input_data = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)
    else:
        prediction = lr_model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("✅ The water is potable (safe to drink).")
    else:
        st.error("⚠️ The water is not potable (unsafe to drink).")