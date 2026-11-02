import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Soil Fertility Prediction",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("soil_fertility_model.pkl")

model = load_model()

with open("feature_names.json", "r") as f:
    features = json.load(f)

st.title("🌱 Soil Fertility Prediction")
st.write(
    "Enter your soil parameters to predict the fertility class "
    "using our Machine Learning model."
)

st.divider()

with st.form("soil_form"):
    values = {}

    for feature in features:
        values[feature] = st.number_input(
            f"{feature}",
            value=0.0,
            format="%.4f"
        )

    submitted = st.form_submit_button("Predict Soil Fertility")

if submitted:
    input_df = pd.DataFrame([values], columns=features)

    try:
        prediction = model.predict(input_df)

        st.success(f"Predicted fertility class: {prediction[0]}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Prediction probabilities")

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability (%)": probabilities * 100
            })

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")