# streamlit_app.py
import pickle
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import requests

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/churn_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model, scaler, feature_names = pickle.load(f)

# -----------------------------
# LOAD REAL LOTTIE ANIMATIONS
# -----------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

happy_anim = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json")  # smiling face
sad_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")    # sad face

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff, #cbebff, #e0f7fa);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #002b36;
    }
    /* Sidebar navigation labels */
    section[data-testid="stSidebar"] .stRadio label span {
        color: #FFD700 !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    /* Sidebar text (Author, Email, GitHub, etc.) */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #FFD700;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FFD700;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #DAA520;
        color: white;
    }

    /* Prediction result cards */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .churn {
        background-color: #ffccd5;
        color: #d00000;
        border: 2px solid #d00000;
    }
    .no-churn {
        background-color: #d8f3dc;
        color: #007200;
        border: 2px solid #007200;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Single Prediction", "📂 Batch Predictions", "📊 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 About the Developer")
st.sidebar.markdown("""
**Sherriff Abdul-Hamid**  
AI Engineer | Data Scientist | Economist  

**Contact:**  
[GitHub](https://github.com/S-ABDUL-AI) | 
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) |  
📧 Sherriffhamid001@gmail.com
""")

# -----------------------------
# HOME
# -----------------------------
if menu == "🏠 Home":
    st.title("⚡ Customer Churn Prediction Dashboard")
    st.markdown("""
        Welcome to the interactive dashboard!  
        Use the sidebar to:
        - 🔮 Predict churn for a **single customer**
        - 📂 Upload a **batch of customers**
        - 📊 Review **model performance**

        This dashboard helps telecom businesses **reduce churn** and improve customer retention.
    """)

    if st.button("🔄 Refresh Home"):
        st.rerun()

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
if menu == "🔮 Single Prediction":
    st.subheader("Predict Churn for a Single Customer")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Customer Details")

        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
        total_charges = st.slider("Total Charges ($)", 0, 9000, 1000)

        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "Contract": contract,
            "PaymentMethod": payment_method,
        }
        input_df = pd.DataFrame([input_dict])

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]
        input_scaled = scaler.transform(input_encoded)

        predict_btn = st.button("🔮 Predict Churn")

    with col2:
        st.markdown("### 🎯 Prediction Results")

        if predict_btn:
            prob = model.predict_proba(input_scaled)[0][1]
            pred = "Churn" if prob >= 0.5 else "No Churn"

            st.write(f"**Probability of Churn:** {prob:.2%}")
            st.write(f"**Classification:** {pred}")

            if prob >= 0.5:
                st.error("⚠️ High risk of churn!")
                st_lottie(sad_anim, height=200, key="sad")
                st.markdown("### 💡 Recommendations")
                st.write("- Offer loyalty discounts 💳")
                st.write("- Improve customer support 📞")
                st.write("- Proactive outreach 📧")
            else:
                st.success("✅ Low risk of churn.")
                st_lottie(happy_anim, height=200, key="happy")
                st.markdown("### 💡 Recommendations")
                st.write("- Keep engagement strong 📰")
                st.write("- Encourage referrals 📈")
                st.write("- Provide loyalty rewards 🎁")

# -----------------------------
# BATCH PREDICTIONS
# -----------------------------
if menu == "📂 Batch Predictions":
    st.subheader("📂 Batch Predictions")

    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_encoded = pd.get_dummies(batch_df, drop_first=True)
        for col in feature_names:
            if col not in batch_encoded.columns:
                batch_encoded[col] = 0
        batch_encoded = batch_encoded[feature_names]
        batch_scaled = scaler.transform(batch_encoded)

        batch_df["Churn_Prob"] = model.predict_proba(batch_scaled)[:, 1]
        batch_df["Prediction"] = (batch_df["Churn_Prob"] >= 0.5).astype(int)

        st.dataframe(batch_df.head())

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

if menu == "📊 Model Performance":
    st.subheader("📊 Model Performance")

    # Simulated performance set (replace with real test set if available)
    X_demo = np.random.rand(100, len(feature_names))
    y_demo = np.random.randint(0, 2, 100)
    y_prob = model.predict_proba(X_demo)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_demo, y_prob)
    roc_auc = auc(fpr, tpr)

    st.markdown("### ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_demo, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

    # Feature Importance (if model supports it)
    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importances")
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        st.bar_chart(importance.set_index("feature"))
