
import streamlit as st
import numpy as np
import joblib

# Set initial configuration
st.set_page_config(page_title="💡 Band Gap Predictor", layout="centered")

# Custom style
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
            padding: 2rem;
        }
        .title {
            color: #0066cc;
        }
        .stButton>button {
            background-color: #0e76a8;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-size: 16px;
        }
        .stNumberInput>div>div>input {
            height: 38px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🔧 Navigation")
pages = ["🏠 Home", "🌲 Random Forest", "⚡ XGBoost", "🌱 Gradient Boosting", "📉 Linear Regression"]
page = st.sidebar.radio("Select a model", pages)

# Common prediction UI
def predict_ui(model_path, scaler_path, r2_score, mse_score):
    st.markdown("### ✍️ Enter Material Parameters")

    # Default inputs set to 0.0
    density = st.number_input("Density (g/cm³)", value=0.0, format="%.5f")
    density_atomic = st.number_input("Atomic Density", value=0.0, format="%.5f")
    formation_energy = st.number_input("Formation Energy per Atom (eV)", value=0.0, format="%.5f")
    energy_above_hull = st.number_input("Energy Above Hull (eV)", value=0.0, format="%.5f")

    if st.button("🔍 Predict Band Gap"):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            input_data = np.array([[density, density_atomic, formation_energy, energy_above_hull]])
            scaled_input = scaler.transform(input_data)
            result = model.predict(scaled_input)[0]
            st.success(f"🔬 Predicted Band Gap: **{result:.4f} eV**")
            st.info(f"📈 R² Score: **{r2_score:.4f}**")
            st.info(f"📉 MSE: **{mse_score:.4f}**")
            st.balloons()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Page Content
if page == "🏠 Home":
    st.markdown("# 🧪 Welcome to the Band Gap Predictor")
    st.markdown("This application predicts the **electronic band gap** of Li–Fe based compounds using various machine learning models.")
    st.markdown("""
        ### 🧠 What it does:
        - Uses 4 simple input parameters:
            - Density
            - Atomic Density
            - Formation Energy
            - Energy Above Hull
        - Predicts the **band gap** using pre-trained ML models
        - Models available:
            - Random Forest 🌲
            - XGBoost ⚡
            - Gradient Boosting 🌱
            - Linear Regression 📉
    """)
    st.image("appimg.png", caption="Concept of Band Gap", use_container_width=True)

elif page == "🌲 Random Forest":
    st.header("🌲 Random Forest Regressor")
    predict_ui("model.pkl", "scaler.pkl", r2_score=0.6135, mse_score=0.7100)

elif page == "⚡ XGBoost":
    st.header("⚡ XGBoost Regressor")
    predict_ui("xgb_model.pkl", "xgb_scaler.pkl", r2_score=0.5903, mse_score=0.7528)

elif page == "🌱 Gradient Boosting":
    st.header("🌱 Gradient Boosting Regressor")
    predict_ui("gbr_model.pkl", "gbr_scaler.pkl", r2_score=0.5733, mse_score=0.7840)

elif page == "📉 Linear Regression":
    st.header("📉 Linear Regression Model")
    predict_ui("linear_regression_model.pkl", "linear_scaler.pkl", r2_score=0.3952, mse_score=1.1112)
