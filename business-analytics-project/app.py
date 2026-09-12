"""
Customer Churn Prediction — Streamlit App
Business decision-support tool: enter a customer's profile and get a churn
risk prediction with a business recommendation.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------
# Load model artifact (model + scaler + encoders, saved together
# in the notebook so preprocessing always matches training exactly)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("model.pkl")

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
label_encoders = artifact["label_encoders"]
feature_columns = artifact["feature_columns"]
numeric_columns = artifact["numeric_columns"]
categorical_columns = artifact["categorical_columns"]
uses_scaled_input = artifact["uses_scaled_input"]
target_mapping_inverse = artifact["target_mapping_inverse"]

st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's profile below to estimate their risk of churning, "
    "so the retention team can prioritize outreach."
)

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        geography = st.selectbox("Geography", label_encoders["Geography"].classes_)
        gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

    with col2:
        balance = st.number_input("Account Balance", min_value=0.0, max_value=260000.0, value=75000.0, step=1000.0)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=80000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Churn Risk")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    raw_input = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    input_df = pd.DataFrame([raw_input])[feature_columns]

    # apply the SAME encoders fit during training
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    model_input = input_df.copy()
    if uses_scaled_input:
        model_input[numeric_columns] = scaler.transform(input_df[numeric_columns])

    prediction = model.predict(model_input)[0]
    probability = model.predict_proba(model_input)[0][1]
    label = target_mapping_inverse[prediction]

    st.subheader("Result")
    if label == "Churned":
        st.error(f"⚠️ High churn risk — predicted: **{label}** (probability: {probability:.1%})")
    else:
        st.success(f"✅ Low churn risk — predicted: **{label}** (probability of churn: {probability:.1%})")

    st.progress(min(max(probability, 0.0), 1.0))

    # ---------------------------------------------------------------
    # Business recommendation, tailored to the risk drivers found
    # in the EDA / feature importance analysis in the notebook
    # ---------------------------------------------------------------
    st.subheader("Business Recommendation")
    reasons = []
    if is_active_member == "No":
        reasons.append("this customer is **inactive**, which roughly doubles churn risk in this dataset")
    if age >= 50:
        reasons.append("**older customers** show a higher churn rate")
    if num_of_products >= 3:
        reasons.append("holding **3+ products** is associated with higher churn, possibly due to product complexity or fees")
    if geography == "Germany":
        reasons.append("customers in **Germany** churn at a structurally higher rate in this dataset")
    if balance >= 100000:
        reasons.append("this customer carries a **high balance**, so losing them has an outsized revenue impact")

    if probability >= 0.5:
        st.write("**Recommended action:** prioritize this customer for a retention call or personalized offer.")
        if reasons:
            st.write("Key risk factors contributing to this prediction:")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.write("No single dominant risk factor — risk is driven by a combination of profile attributes.")
    else:
        st.write(
            "**Recommended action:** no urgent intervention needed. Continue standard engagement; "
            "revisit if activity drops or product usage changes."
        )

st.divider()
st.caption(
    "Model: trained on historical customer data. Predictions are probabilistic estimates "
    "to support — not replace — retention team judgment."
)
