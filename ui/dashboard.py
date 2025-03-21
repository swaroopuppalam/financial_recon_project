import streamlit as st
import requests

st.title("Financial Reconciliation - Anomaly Detection")

company = st.text_input("Company")
account = st.text_input("Account")
balance_diff = st.number_input("Balance Difference")

if st.button("Detect Anomaly"):
    response = requests.post("http://backend:8000/detect_anomaly/", json={
        "Company": company,
        "Account": account,
        "Balance Difference": balance_diff
    })
    result = response.json()
    st.write(f"Anomaly: {result['Anomaly']}")
