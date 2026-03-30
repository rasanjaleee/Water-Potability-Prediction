import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AquaSense · Water Potability",
    page_icon="💧",
    layout="wide"
)

# ── Title ───────────────────────────────────────────────────
st.title("💧 AquaSense - Water Potability Analyzer")
st.markdown("AI-powered system to predict if water is **safe for drinking**.")

# ── Load Models ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load pre-trained fitted models"""
    with open("rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    return rf, lr

try:
    rf_model, lr_model = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ── Layout ─────────────────────────────────────────────────
left, right = st.columns([2, 1])

# ==========================================================
# LEFT SIDE
# ==========================================================
with left:

    st.subheader("🧪 Enter Water Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        pH = st.number_input("pH", 0.0, 14.0, 7.0)
        Hardness = st.number_input("Hardness (mg/L)", 0.0, 150.0)
        Solids = st.number_input("Solids (mg/L)", 0.0, 5000.0)

    with col2:
        Chloramines = st.number_input("Chloramines (mg/L)", 0.0, 5.0)
        Sulfate = st.number_input("Sulfate (mg/L)", 0.0, 300.0)
        Conductivity = st.number_input("Conductivity (µS/cm)", 0.0, 400.0)

    with col3:
        Organic_carbon = st.number_input("Organic Carbon (mg/L)", 0.0, 10.0)
        Trihalomethanes = st.number_input("Trihalomethanes (µg/L)", 0.0, 80.0)
        Turbidity = st.number_input("Turbidity (NTU)", 0.0, 3.0)

    # ── Model Selection ───────────────────────────────────
    model_choice = st.radio(
        "🤖 Select Model",
        ["Random Forest", "Logistic Regression"],
        horizontal=True
    )

    # ── Validation Warnings ───────────────────────────────
    warnings = []
    if pH < 4 or pH > 10:
        warnings.append("⚠️ Extreme pH value detected")
    if Solids > 10000:
        warnings.append("⚠️ Very high dissolved solids")
    if Turbidity > 5:
        warnings.append("⚠️ High turbidity may indicate contamination")
    for w in warnings:
        st.warning(w)

    # ── Prediction Button ─────────────────────────────────
    if st.button("⚗️ Analyze Water"):

        if not models_loaded:
            st.error("Models not found. Please add rf_model.pkl and lr_model.pkl")
        else:
            # Prepare input data as array
            input_data = np.array([[pH, Hardness, Solids, Chloramines,
                                    Sulfate, Conductivity,
                                    Organic_carbon, Trihalomethanes, Turbidity]])

            # Select model
            model = rf_model if model_choice == "Random Forest" else lr_model

            # Predict
            prediction = model.predict(input_data)[0]

            # Confidence (if available)
            try:
                prob = model.predict_proba(input_data)[0]
                confidence = max(prob) * 100
                safe_prob = prob[1] * 100
                unsafe_prob = prob[0] * 100
            except:
                confidence = None

            # ── Result Display ──────────────────────────
            st.markdown("---")
            if prediction == 1:
                st.success("✅ Water is SAFE for drinking")
            else:
                st.error("⚠️ Water is NOT SAFE for drinking")

            # ── Probability Breakdown ───────────────────
            if confidence:
                st.write(f"🔍 Confidence: **{confidence:.2f}%**")
                st.write(f"🟢 Safe: {safe_prob:.2f}% | 🔴 Unsafe: {unsafe_prob:.2f}%")

            # ── Recommendations ─────────────────────────
            if prediction == 0:
                st.info("""💡 **Recommended Actions:** 
                - Boil water before drinking  
                - Use filtration systems (RO/UV)  
                - Reduce turbidity via sedimentation  
                - Test water in a certified laboratory""")

            # ── Feature Importance (RF only) ────────────
            if model_choice == "Random Forest":
                st.subheader("📊 Feature Importance")
                features = ["pH", "Hardness", "Solids", "Chloramines",
                            "Sulfate", "Conductivity", "Organic Carbon",
                            "Trihalomethanes", "Turbidity"]
                try:
                    importance = model.feature_importances_
                    df_imp = pd.DataFrame({
                        "Feature": features,
                        "Importance": importance
                    }).sort_values(by="Importance", ascending=False)
                    st.bar_chart(df_imp.set_index("Feature"))
                except:
                    st.warning("Feature importance not available")

# ==========================================================
# RIGHT SIDE
# ==========================================================
with right:
    st.subheader("📋 WHO Guidelines")
    st.table(pd.DataFrame({
        "Parameter": ["pH", "Hardness", "Solids", "Chloramines",
                      "Sulfate", "Conductivity", "Organic Carbon",
                      "Trihalomethanes", "Turbidity"],
        "Safe Range": ["6.5–8.5", "<300", "<500", "<4",
                       "<250", "<400", "<2", "<80", "<5"]
    }))
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    - Use lab-tested data for accuracy  
    - High TDS doesn't always mean unsafe  
    - Random Forest usually performs better  
    - Always verify with official water testing  
    """)
    st.markdown("---")
    st.markdown("🌊 **AquaSense v2.0**  Powered by Machine Learning  _For educational use only_")