import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("best_loan_model.pkl")

# ---------------- UI Style ----------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    height: 45px;
    font-size: 18px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Loan Default Prediction System")
st.markdown("### Enter Applicant Details")

# ---------------- Layout (Side by Side) ----------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 100, 30)
    Income = st.number_input("Income", 0, 1000000, 50000)
    LoanAmount = st.number_input("Loan Amount", 0, 500000, 20000)
    CreditScore = st.number_input("Credit Score", 300, 850, 650)
    MonthsEmployed = st.number_input("Months Employed", 0, 500, 12)
    NumCreditLines = st.number_input("Number of Credit Lines", 0, 20, 2)
    InterestRate = st.number_input("Interest Rate", 0.0, 50.0, 10.0)

with col2:
    LoanTerm = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    DTIRatio = st.number_input("DTI Ratio", 0.0, 1.0, 0.3)
    HasMortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
    HasDependents = st.selectbox("Has Dependents", ["No", "Yes"])
    HasCoSigner = st.selectbox("Has Co-Signer", ["No", "Yes"])
    Education = st.selectbox(
        "Education",
        ["High School", "Bachelor's", "Master's", "PhD"]
    )
    EmploymentType = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    )

col3, col4 = st.columns(2)

with col3:
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"]
    )

with col4:
    LoanPurpose = st.selectbox(
        "Loan Purpose",
        ["Business", "Education", "Home", "Other"]
    )

st.markdown("---")

# ---------------- Prediction ----------------
if st.button("Predict Loan Risk"):

    # Base numeric data
    data = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "HasMortgage": 1 if HasMortgage == "Yes" else 0,
        "HasDependents": 1 if HasDependents == "Yes" else 0,
        "HasCoSigner": 1 if HasCoSigner == "Yes" else 0,
    }

    # One-hot columns (same as training)
    one_hot = {
        # Education
        "Education_Bachelor's": 0,
        "Education_High School": 0,
        "Education_Master's": 0,
        "Education_PhD": 0,

        # Employment (Full-time is base)
        "EmploymentType_Part-time": 0,
        "EmploymentType_Self-employed": 0,
        "EmploymentType_Unemployed": 0,

        # Marital (Divorced is base)
        "Marital_Married": 0,
        "Marital_Single": 0,

        # Purpose
        "Purpose_Business": 0,
        "Purpose_Education": 0,
        "Purpose_Home": 0,
        "Purpose_Other": 0
    }

    # Set selected = 1
    one_hot[f"Education_{Education}"] = 1

    if EmploymentType != "Full-time":
        one_hot[f"EmploymentType_{EmploymentType}"] = 1

    if MaritalStatus != "Divorced":
        one_hot[f"Marital_{MaritalStatus}"] = 1

    one_hot[f"Purpose_{LoanPurpose}"] = 1

    # Merge
    data.update(one_hot)

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ---------------- Result ----------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk of Default")
        st.write(f"Default Probability: **{probability:.2%}**")
    else:
        st.success("Low Risk Applicant")
        st.write(f"Default Probability: **{probability:.2%}**")
