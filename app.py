import json
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Soil Fertility AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F7F0;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* All headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #174D35 !important;
    -webkit-text-fill-color: #174D35 !important;
}

/* Normal text */
.stApp p,
.stApp label,
.stApp .stMarkdown {
    color: #344A3D;
}

/* Header card */
.hero {
    background: linear-gradient(120deg, #174D35, #367B50);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    -webkit-text-fill-color: white !important;
    font-size: 38px;
}

.hero p {
    color: #FFFFFF !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EAF1E8;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #174D35 !important;
    -webkit-text-fill-color: #174D35 !important;
}

/* Input fields */
[data-testid="stNumberInput"] input {
    color: #263B30 !important;
    background-color: #FFFFFF !important;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #236B45 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}

div.stButton > button p,
div.stFormSubmitButton > button p {
    color: white !important;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #174D35 !important;
}

/* Result card */
.result-card {
    background: linear-gradient(120deg, #174D35, #367B50);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
}

.result-card h2,
.result-card p {
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Dividers */
hr {
    border-color: #DCE6DA;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("soil_fertility_model.pkl")


model = load_model()

with open("feature_names.json", "r") as f:
    features = json.load(f)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🌱 Soil Fertility AI")

    st.write("### About the Project")

    st.write(
        "This application uses Machine Learning "
        "to predict soil fertility based on "
        "essential soil nutrients and properties."
    )

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Input Features:** 12 Soil Parameters")

    st.write("**Output:** Soil Fertility Class")

    st.divider()

    st.caption("Developed as a Machine Learning Project")


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🌱 Soil Fertility Prediction</h1>
    <p>
        Intelligent Soil Analysis Using Machine Learning
        <br>
        Analyse soil parameters and predict fertility levels.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT SECTION ----------------

st.markdown("## 🧪 Enter Soil Parameters")

st.write(
    "Provide the following soil measurements "
    "to generate a fertility prediction."
)

# Sample values for quick testing
sample_values = {
    "N": 138,
    "P": 8.6,
    "K": 560,
    "pH": 7.46,
    "EC": 0.62,
    "OC": 0.70,
    "S": 5.9,
    "Zn": 0.24,
    "Fe": 0.31,
    "Cu": 0.77,
    "Mn": 8.71,
    "B": 0.11
}

if st.button("📋 Load Sample Soil Data"):
    for feature in features:
        st.session_state[f"input_{feature}"] = float(
            sample_values.get(feature, 0.0)
        )

    st.rerun()


# Organise inputs into categories
nutrients = ["N", "P", "K", "S"]
properties = ["pH", "EC", "OC"]
micronutrients = ["Zn", "Fe", "Cu", "Mn", "B"]

input_values = {}

with st.form("soil_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌿 Primary Nutrients")

        for feature in nutrients:
            if feature in features:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(st.session_state.get(
                        f"input_{feature}", 0.0
                    )),
                    format="%.4f",
                    key=f"input_{feature}"
                )

    with col2:
        st.markdown("### 🌍 Soil Properties")

        for feature in properties:
            if feature in features:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(st.session_state.get(
                        f"input_{feature}", 0.0
                    )),
                    format="%.4f",
                    key=f"input_{feature}"
                )

    with col3:
        st.markdown("### 🔬 Micronutrients")

        for feature in micronutrients:
            if feature in features:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(st.session_state.get(
                        f"input_{feature}", 0.0
                    )),
                    format="%.4f",
                    key=f"input_{feature}"
                )

    # Any additional features will still be included
    remaining_features = [
        f for f in features
        if f not in nutrients + properties + micronutrients
    ]

    for feature in remaining_features:
        input_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.4f",
            key=f"input_{feature}"
        )

    st.write("")

    submitted = st.form_submit_button(
        "🌱 Predict Soil Fertility",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------

if submitted:

    input_df = pd.DataFrame(
        [[input_values[f] for f in features]],
        columns=features
    )

    try:
        prediction = model.predict(input_df)[0]

        st.divider()

        st.markdown("## 📊 Soil Analysis Results")

        st.markdown(f"""
        <div class="result-card">
            <p>YOUR PREDICTED FERTILITY CLASS</p>
            <h2>Class {prediction}</h2>
            <p>Prediction generated successfully</p>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in model.classes_],
                "Probability (%)": probabilities * 100
            })

            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:

                st.markdown("### 📈 Prediction Probabilities")

                chart_df = probability_df.set_index("Class")

                st.bar_chart(
                    chart_df,
                    horizontal=True,
                    x_label="Probability (%)"
                )

            with result_col2:

                st.markdown("### 📋 Probability Details")

                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True
                )

                max_probability = max(probabilities) * 100

                st.metric(
                    "Highest Class Probability",
                    f"{max_probability:.2f}%"
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <p style="color:#607568;">
        🌱 Soil Fertility Prediction System
        <br>
        Powered by Machine Learning | Random Forest Classifier
    </p>
</div>
""", unsafe_allow_html=True)
