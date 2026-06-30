# 🏦 Loan Approval Prediction System

This is a Machine Learning-based web application that predicts whether a loan will be approved or rejected based on applicant details.

## 🚀 Project Overview
The system takes user inputs like income, CIBIL score, loan amount, assets, etc., and predicts loan approval using trained ML models.

The app is built using **Streamlit**, which provides an interactive web interface directly from Python.

---

# Folder Structure
`
Loan-Approval-Prediction-System
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│   ├── test_data.csv
│
├── preprocessing/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│
├── results/
│   ├── model_metrics.xlsx
│   ├── classification_report.txt
│
├── streamlit_app/
│   └── app.py
│
├── training/
│   └── main.py
│
├── requirements.txt
└── README.md
`

## 🧠 Machine Learning Models Used
- Logistic Regression
- Decision Tree
- Random Forest

---

## 🛠️ Technologies Used
- Python
- Streamlit (UI)
- Scikit-learn
- Pandas
- Joblib

---

## 📥 Input Features
- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets
- Commercial Assets
- Luxury Assets
- Bank Assets

---

## ▶️ How to Run the Project

1. Open terminal in project folder  
2. Install dependencies:

pip install streamlit pandas scikit-learn joblib

3. Run the app:

streamlit run app.py

---

## 🎯 Output
- 🎉 Loan Approved
- ❌ Loan Rejected

---

## 📌 Author
Name : Samruddhi Panhlkar
