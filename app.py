import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/diabetes_model.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, features

model, features = load_model()

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "An educational machine learning application "
    "that estimates a diabetes-related classification "
    "from health measurements."
)

st.warning(
    "This is an educational prototype, not a diagnostic tool. "
    "Its predictions are not clinically validated and must not "
    "be used to make medical decisions."
)

st.subheader("Enter the health measurements")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0, max_value=20, value=1
        )
        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0, max_value=400, value=120
        )
        blood_pressure = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0, max_value=200, value=70
        )
        skin_thickness = st.number_input(
            "Skin thickness (mm)",
            min_value=0, max_value=100, value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin (dataset measurement)",
            min_value=0, max_value=1000, value=80
        )
        bmi = st.number_input(
            "BMI",
            min_value=0.0, max_value=100.0,
            value=25.0, step=0.1
        )
        dpf = st.number_input(
            "Diabetes pedigree function",
            min_value=0.0, max_value=5.0,
            value=0.5, step=0.01
        )
        age = st.number_input(
            "Age (years)",
            min_value=1, max_value=120, value=30
        )

    submitted = st.form_submit_button(
        "Predict", use_container_width=True
    )

if submitted:
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    input_df = pd.DataFrame([values])[features]

    # Avoid making a misleading prediction from
    # clearly absent or invalid key measurements.
    if glucose <= 0 or bmi <= 0 or blood_pressure <= 0:
        st.error(
            "Enter valid, positive glucose, BMI and "
            "blood pressure measurements before predicting."
        )
    else:
        prediction = int(model.predict(input_df)[0])
        score = float(model.predict_proba(input_df)[0][1])

        st.subheader("Model output")

        if prediction == 1:
            st.error(
                "The model classified this input as "
                "Outcome 1 (positive in the training dataset)."
            )
        else:
            st.success(
                "The model classified this input as "
                "Outcome 0 (negative in the training dataset)."
            )

        st.metric(
            "Model-estimated probability for Outcome 1",
            f"{score:.1%}"
        )

        st.caption(
            "This probability is not a clinically validated "
            "personal diabetes-risk estimate. A prediction "
            "cannot diagnose or rule out diabetes."
        )
