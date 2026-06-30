import streamlit as st
import joblib
import pandas as pd

# Load model
# Select Model
model_name = st.selectbox(
    "Select Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    )
)

# Load selected model
if model_name == "Logistic Regression":
    model = joblib.load("logistic_regression.pkl")

elif model_name == "Decision Tree":
    model = joblib.load("decision_tree.pkl")

else:
    model = joblib.load("random_forest.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("🏦 Loan Approval Prediction System")

st.write("Enter the applicant details below.")

# Number of Dependents
dependents = st.number_input("Number of Dependents", min_value=0)

# Annual Income
income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
# Loan Term (in years)
loan_term = st.number_input("Loan Term (Years)", min_value=1)
# Residential Assets Value
residential_assets = st.number_input("Residential Assets Value", min_value=0)
# Commercial Assets Value
commercial_assets = st.number_input("Commercial Assets Value", min_value=0)
# Luxury Assets Value
luxury_assets = st.number_input("Luxury Assets Value", min_value=0)
# Bank Assets Value
bank_assets = st.number_input("Bank Assets Value", min_value=0)

# CIBIL Score
cibil = st.number_input("CIBIL Score", min_value=300, max_value=900)
# Education
education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)
# Self Employed
self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

st.write("Dependents:", dependents)
st.write("Income:", income)
st.write("Loan Amount:", loan_amount)
st.write("Loan Term:", loan_term)
st.write("Residential Assets Value:", residential_assets)
st.write("Commercial Assets Value:", commercial_assets)
st.write("Luxury Assets Value:", luxury_assets)
st.write("Bank Assets Value:", bank_assets)
st.write("CIBIL Score:", cibil)
st.write("Education:", education)
st.write("Self Employed:", self_employed)
if st.button("Predict Loan Approval"):

    # Convert categorical values
    education_value = 1 if education == "Graduate" else 0
    self_employed_value = 1 if self_employed == "Yes" else 0

    # Create input data in the same order as the model was trained
    input_data = [[
        dependents,
        education_value,
        self_employed_value,
        income,
        loan_amount,
        loan_term,
        cibil,
        residential_assets,
        commercial_assets,
        luxury_assets,
        bank_assets
    ]]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🎉 Loan Approved")
    else:
        st.error("❌ Loan Rejected")