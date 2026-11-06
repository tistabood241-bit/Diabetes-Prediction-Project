
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# Custom design
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8fbff);
}
.main-title {
    text-align: center;
    color: #123b67;
    font-size: 36px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #52677d;
    font-size: 16px;
    margin-bottom: 25px;
}
div.stButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 10px;
    height: 48px;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model and scaler
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_files():
    model = load_model(BASE_DIR / "diabetes_model_final.keras")

    with open(BASE_DIR / "diabetes_scaler_final.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler

try:
    model, scaler = load_files()
except Exception as error:
    st.error("Could not load the model or scaler. Please check that both files are in the repository.")
    st.stop()

# --------------------------------------------------
# Website header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🩺 Diabetes Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Early Diabetes Detection Using Artificial Neural Network</div>',
    unsafe_allow_html=True
)

st.info(
    "Please answer all questions based on your current condition."
)

# --------------------------------------------------
# Input form
# --------------------------------------------------
with st.form("diabetes_form"):

    st.subheader("Personal Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    st.divider()
    st.subheader("Symptoms")

    polyuria = st.selectbox(
        "Do you experience excessive urination (Polyuria)?",
        ["No", "Yes"]
    )

    polydipsia = st.selectbox(
        "Do you experience excessive thirst (Polydipsia)?",
        ["No", "Yes"]
    )

    sudden_weight_loss = st.selectbox(
        "Have you experienced sudden weight loss?",
        ["No", "Yes"]
    )

    weakness = st.selectbox(
        "Do you experience general weakness?",
        ["No", "Yes"]
    )

    polyphagia = st.selectbox(
        "Do you experience excessive hunger (Polyphagia)?",
        ["No", "Yes"]
    )

    genital_thrush = st.selectbox(
        "Have you experienced genital thrush?",
        ["No", "Yes"]
    )

    visual_blurring = st.selectbox(
        "Do you experience blurred vision?",
        ["No", "Yes"]
    )

    itching = st.selectbox(
        "Do you experience itching?",
        ["No", "Yes"]
    )

    irritability = st.selectbox(
        "Do you experience irritability?",
        ["No", "Yes"]
    )

    delayed_healing = st.selectbox(
        "Do your wounds take longer than usual to heal?",
        ["No", "Yes"]
    )

    partial_paresis = st.selectbox(
        "Do you experience partial muscle weakness (Partial Paresis)?",
        ["No", "Yes"]
    )

    muscle_stiffness = st.selectbox(
        "Do you experience muscle stiffness?",
        ["No", "Yes"]
    )

    alopecia = st.selectbox(
        "Have you experienced unusual hair loss (Alopecia)?",
        ["No", "Yes"]
    )

    obesity = st.selectbox(
        "Do you have obesity?",
        ["No", "Yes"]
    )

    submitted = st.form_submit_button("Predict Diabetes Risk")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    # Keep exactly the same column names and order
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Polyuria": 1 if polyuria == "Yes" else 0,
        "Polydipsia": 1 if polydipsia == "Yes" else 0,
        "sudden weight loss": 1 if sudden_weight_loss == "Yes" else 0,
        "weakness": 1 if weakness == "Yes" else 0,
        "Polyphagia": 1 if polyphagia == "Yes" else 0,
        "Genital thrush": 1 if genital_thrush == "Yes" else 0,
        "visual blurring": 1 if visual_blurring == "Yes" else 0,
        "Itching": 1 if itching == "Yes" else 0,
        "Irritability": 1 if irritability == "Yes" else 0,
        "delayed healing": 1 if delayed_healing == "Yes" else 0,
        "partial paresis": 1 if partial_paresis == "Yes" else 0,
        "muscle stiffness": 1 if muscle_stiffness == "Yes" else 0,
        "Alopecia": 1 if alopecia == "Yes" else 0,
        "Obesity": 1 if obesity == "Yes" else 0
    }])

    try:
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(
            scaled_data,
            verbose=0
        )

        probability = float(np.asarray(prediction).reshape(-1)[0])

        st.divider()
        st.subheader("Prediction Result")

        if probability >= 0.5:
            st.error("The model indicates a higher likelihood of diabetes.")
        else:
            st.success("The model indicates a lower likelihood of diabetes.")

        st.metric(
            "Model output",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This percentage is the model's output, not a clinically validated "
            "probability of having diabetes."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")

# --------------------------------------------------
# Medical disclaimer
# --------------------------------------------------
st.divider()

st.warning(
    "**Medical disclaimer:** This website is an educational screening project "
    "and is not a medical diagnostic tool. Its result cannot confirm or rule "
    "out diabetes. Please consult a qualified healthcare professional for "
    "appropriate testing and interpretation."
)
