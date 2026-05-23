import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ================= Load Model + Preprocess =================
model = joblib.load('model.pkl')
preprocess = joblib.load('preprocess.pkl')

# ================= Input Function =================
def getInput():

    Age = st.slider('Age', min_value=18, max_value=100, step=1)

    Income = st.slider('Income', min_value=0, max_value=200000, step=1000)

    LoanAmount = st.slider('Loan Amount', min_value=0, max_value=100000, step=1000)

    CreditScore = st.slider('Credit Score', min_value=300, max_value=900, step=1)

    YearsExperience = st.slider('Years of Experience', min_value=0, max_value=50, step=1)

    employment_options = preprocess.named_transformers_['cat'].categories_[0]

    EmploymentType = st.selectbox(
        "Employment Type",
        employment_options)

    return pd.DataFrame(
        data=[[
            Age,
            Income,
            LoanAmount,
            CreditScore,
            YearsExperience,
            EmploymentType
        ]],
        columns=[
            'Age',
            'Income',
            'LoanAmount',
            'CreditScore',
            'YearsExperience',
            'EmploymentType'
        ]
    )

# ================= UI =================
st.title("💰 Loan Approval Predictor")

test = getInput()

st.subheader("📊 Input Data")
st.dataframe(test)

# ================= Prediction =================
if st.button("Predict Loan Status"):

    # preprocess (VERY IMPORTANT)
    processed = preprocess.transform(test)

    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📊 Approval Probability: {proba:.2f}")
