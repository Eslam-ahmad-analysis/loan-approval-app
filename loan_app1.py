import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ================= Load =================
model = joblib.load('model.pkl')
preprocess = joblib.load('preprocess.pkl')

# ================= Page Config =================
st.set_page_config(
    page_title="Loan Dashboard",
    page_icon="💰",
    layout="wide"
)

# ================= Title =================
st.title("💰 Loan Approval Dashboard")
st.markdown("### Predict loan approval using Machine Learning")

# ================= Sidebar Inputs =================
st.sidebar.header("📌 Input Features")

Age = st.sidebar.slider("Age", 18, 100, 30)
Income = st.sidebar.slider("Income", 0, 200000, 50000)
LoanAmount = st.sidebar.slider("Loan Amount", 0, 100000, 10000)
CreditScore = st.sidebar.slider("Credit Score", 300, 900, 700)
YearsExperience = st.sidebar.slider("Years of Experience", 0, 50, 5)

employment_options = preprocess.named_transformers_['cat'].categories_[0]

EmploymentType = st.sidebar.selectbox(
    "Employment Type",
    employment_options
)

# ================= Input DF =================
input_df = pd.DataFrame([[
    Age, Income, LoanAmount, CreditScore, YearsExperience, EmploymentType
]],
columns=[
    "Age", "Income", "LoanAmount",
    "CreditScore", "YearsExperience",
    "EmploymentType"
])

# ================= Main Layout =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction")

    if st.button("🚀 Predict Loan Status"):

        processed = preprocess.transform(input_df)

        prediction = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0][1]

        # ================= Result Card =================
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # ================= Probability =================
        st.metric(label="Approval Probability", value=f"{proba:.2f}")

        st.progress(float(proba))

        # ================= Insight =================
        if proba > 0.7:
            st.info("🟢 High chance of approval")
        elif proba > 0.4:
            st.warning("🟡 Medium chance")
        else:
            st.error("🔴 Low chance")
