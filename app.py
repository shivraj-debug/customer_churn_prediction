import streamlit as st
import pickle
import numpy as np

# load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details:")

# INPUTS
credit_score = st.number_input("Credit Score", 300, 900)
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure (years)", 0, 10)
balance = st.number_input("Balance", 0.0)
num_products = st.number_input("Number of Products", 1, 4)
has_card = st.selectbox("Has Credit Card", [0,1])
is_active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Estimated Salary", 0.0)

# categorical encoding (same as your colab)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])

geo_map = {"France":0, "Spain":1, "Germany":2}
gender_map = {"Male":1, "Female":0}

geo = geo_map[geography]
gen = gender_map[gender]

# SCALE (same columns you used)
scaled = scaler.transform([[credit_score, balance, salary]])

# FINAL INPUT
input_data = np.array([[credit_score, geo, gen, age, tenure,
                        balance, num_products, has_card,
                        is_active, salary]])

if st.button("Predict"):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"Churn Probability: {prob:.2f}")

    if prob > 0.7:
        st.warning("⚠️ High Risk Customer")
    elif prob > 0.4:
        st.info("⚡ Medium Risk")
    else:
        st.success("✅ Low Risk")

    if result == 1:
        st.error("❌ Customer will CHURN")
    else:
        st.success("✅ Customer will NOT churn")