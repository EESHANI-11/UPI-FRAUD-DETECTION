import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("upi_fraud_detection.pkl", 'rb') as file:
    model = pickle.load(file)

# --- Custom CSS for Dark Theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0e1117;
        color: white;
    }
    h2, h3, h4 { color: #00FFD1; }
    .stButton > button {
        background-color: #00FFD1;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #FF007F;
        color: white;
    }
    .stFileUploader {
        border: 2px dashed #00FFD1 !important;
        background-color: #1E1E1E !important;
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #333; }
    ::-webkit-scrollbar-thumb { background: #00FFD1; border-radius: 10px; }
    .stSuccess {
        color: #00FFD1;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    .stError {
        color: #FF007F;
        font-weight: bold;
        text-align: center;
        font-size: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Page Title
st.markdown("<h2 style='text-align: center;'>🚀 UPI Transaction Fraud Detector</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check for fraud in transactions using AI-powered predictions</p>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Enter Transaction Details")

amount = st.sidebar.number_input("Enter transaction amount (INR)", min_value=0.0, max_value=500000.0, step=10.0, format="%.2f")
year = st.sidebar.number_input("Transaction Year", min_value=2000, max_value=2100, step=1)
month = st.sidebar.selectbox("Transaction Month", list(range(1, 13)))

transaction_type = st.sidebar.selectbox("Select Transaction Type", ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"])
payment_gateway = st.sidebar.selectbox("Select Payment Gateway", ["Bank of Data", "CReditPAY", "Dummy Bank", "Gamma Bank", "Other", "SamplePay", "Sigma Bank", "UPI Pay"])
transaction_state = st.sidebar.selectbox("Select Transaction State", [
    "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", 
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", 
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
    "Uttarakhand", "West Bengal"
])
merchant_category = st.sidebar.selectbox("Select Merchant Category", [
    "Donations and Devotion", "Financial services and Taxes", "Home delivery", "Investment", "More Services", 
    "Other", "Purchases", "Travel bookings", "Utilities"
])

# --- Convert Inputs to Model Feature Format ---
feature_dict = {
    "amount": amount,
    "Year": year,
    "Month": month,
    
    # Transaction Types
    "Transaction_Type_Bill Payment": 1 if transaction_type == "Bill Payment" else 0,
    "Transaction_Type_Investment": 1 if transaction_type == "Investment" else 0,
    "Transaction_Type_Other": 1 if transaction_type == "Other" else 0,
    "Transaction_Type_Purchase": 1 if transaction_type == "Purchase" else 0,
    "Transaction_Type_Refund": 1 if transaction_type == "Refund" else 0,
    "Transaction_Type_Subscription": 1 if transaction_type == "Subscription" else 0,

    # Payment Gateways
    "Payment_Gateway_Bank of Data": 1 if payment_gateway == "Bank of Data" else 0,
    "Payment_Gateway_CReditPAY": 1 if payment_gateway == "CReditPAY" else 0,
    "Payment_Gateway_Dummy Bank": 1 if payment_gateway == "Dummy Bank" else 0,
    "Payment_Gateway_Gamma Bank": 1 if payment_gateway == "Gamma Bank" else 0,
    "Payment_Gateway_Other": 1 if payment_gateway == "Other" else 0,
    "Payment_Gateway_SamplePay": 1 if payment_gateway == "SamplePay" else 0,
    "Payment_Gateway_Sigma Bank": 1 if payment_gateway == "Sigma Bank" else 0,
    "Payment_Gateway_UPI Pay": 1 if payment_gateway == "UPI Pay" else 0,
}

# Add all transaction states dynamically
for state in [
    "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
]:
    feature_dict[f"Transaction_State_{state}"] = 1 if transaction_state == state else 0

# Merchant Categories
for category in [
    "Donations and Devotion", "Financial services and Taxes", "Home delivery", "Investment", "More Services", 
    "Other", "Purchases", "Travel bookings", "Utilities"
]:
    feature_dict[f"Merchant_Category_{category}"] = 1 if merchant_category == category else 0

df = pd.DataFrame([feature_dict])

# --- Fraud Detection ---
if st.sidebar.button("🚨 Check Transaction"):
    prediction = model.predict(df)
    
    if prediction[0] == 1:
        st.sidebar.markdown("<p class='stError'>🚨 **Fraudulent Transaction Detected!**</p>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<p class='stSuccess'>✅ **Transaction is Safe!**</p>", unsafe_allow_html=True)

# --- Bulk CSV Upload ---
st.subheader("📂 Upload CSV to Check Multiple Transactions")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    predictions = model.predict(data)
    data["Fraud Status"] = ["🚨 Fraud" if pred == 1 else "✅ Safe" for pred in predictions]
    
    st.write("### 📊 Fraud Detection Results")
    st.dataframe(data)
