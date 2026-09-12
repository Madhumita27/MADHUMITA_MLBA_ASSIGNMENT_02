"""
Customer Churn Prediction — Streamlit Business Decision-Support App

Three pages required for the assignment:
1. Business Problem
2. Data Insights
3. Prediction
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ---------------------------------------------------------------
# File paths
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")


# ---------------------------------------------------------------
# Load model
# ---------------------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------
# Load dataset for Page 2
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("📉 Customer Churn")
st.sidebar.write("Business Decision-Support Application")

page = st.sidebar.radio(
    "Select a page",
    [
        "🏠 Business Problem",
        "📊 Data Insights",
        "🔮 Prediction"
    ]
)


# ===============================================================
# PAGE 1 — BUSINESS PROBLEM
# ===============================================================
if page == "🏠 Business Problem":

    st.title("📉 Customer Churn Prediction")

    st.subheader("Business Problem")
    st.write(
        "A retail bank is losing customers at a meaningful rate. Customer churn "
        "reduces recurring revenue and increases the cost of acquiring replacement "
        "customers. The bank needs a way to identify customers who are likely to "
        "leave before they actually churn."
    )

    st.subheader("Analytics Objective")
    st.write(
        "The objective is to predict the probability that a customer will churn "
        "and help the retention team prioritize proactive outreach to high-risk "
        "customers under a limited retention budget."
    )

    st.subheader("Target Variable")
    st.info("Exited: 1 = Churned, 0 = Retained")

    st.subheader("Dataset Information")

    try:
        df = load_data()

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Customers", f"{len(df):,}")
        col2.metric("Number of Variables", f"{df.shape[1]:,}")
        col3.metric("Overall Churn Rate", f"{df['Exited'].mean():.1%}")

        st.write(
            "The dataset contains customer demographic, account and engagement "
            "information such as credit score, geography, gender, age, tenure, "
            "balance, number of products, credit-card ownership, active-member "
            "status and estimated salary."
        )

        with st.expander("View Dataset Columns"):
            st.write(list(df.columns))

    except Exception as e:
        st.warning("Dataset could not be loaded for the information section.")
        st.caption(str(e))

    st.subheader("Project Flow")
    st.write(
        "Business Problem → Dataset → Data Preparation → EDA → ML Model → "
        "Evaluation → Prediction → Business Recommendation → Streamlit → Deployment"
    )


# ===============================================================
# PAGE 2 — DATA INSIGHTS
# ===============================================================
elif page == "📊 Data Insights":

    st.title("📊 Data Insights")

    try:
        df = load_data()

        st.subheader("Key Statistics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", f"{len(df):,}")
        col2.metric("Churned", f"{int(df['Exited'].sum()):,}")
        col3.metric("Retained", f"{int((df['Exited'] == 0).sum()):,}")
        col4.metric("Churn Rate", f"{df['Exited'].mean():.1%}")

        # -------------------------------------------------------
        # Chart 1: Churn by Geography
        # -------------------------------------------------------
        st.subheader("1. Churn Rate by Geography")

        geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        st.bar_chart(geo_churn)

        st.write(
            "Business insight: Germany shows a noticeably higher churn rate "
            "than France and Spain, suggesting a need to investigate "
            "region-specific retention issues."
        )

        # -------------------------------------------------------
        # Chart 2: Churn by Active Member
        # -------------------------------------------------------
        st.subheader("2. Churn Rate by Active Member Status")

        active_churn = (
            df.groupby("IsActiveMember")["Exited"]
            .mean()
            .rename(index={0: "Inactive", 1: "Active"})
        )
        st.bar_chart(active_churn)

        st.write(
            "Business insight: Inactive customers have a substantially higher "
            "churn rate than active customers. Engagement campaigns can therefore "
            "be an actionable retention strategy."
        )

        # -------------------------------------------------------
        # Chart 3: Churn by Number of Products
        # -------------------------------------------------------
        st.subheader("3. Churn Rate by Number of Products")

        product_churn = df.groupby("NumOfProducts")["Exited"].mean()
        st.bar_chart(product_churn)

        st.write(
            "Business insight: Customers with 3 or more products show higher churn "
            "in this dataset. The bank should review product complexity, fees and "
            "the customer experience for these customers."
        )

        # -------------------------------------------------------
        # Chart 4: Churn by Age Group
        # -------------------------------------------------------
        st.subheader("4. Churn Rate by Age Group")

        df_insight = df.copy()
        df_insight["AgeGroup"] = pd.cut(
            df_insight["Age"],
            bins=[17, 30, 40, 50, 60, 100],
            labels=["18–30", "31–40", "41–50", "51–60", "61+"]
        )
        age_churn = df_insight.groupby("AgeGroup", observed=False)["Exited"].mean()
        st.bar_chart(age_churn)

        st.write(
            "Business insight: Churn is higher among older customers. "
            "Retention teams can consider personalized engagement for this segment."
        )

        # -------------------------------------------------------
        # Summary
        # -------------------------------------------------------
        st.subheader("Overall Business Insights")

        st.markdown(
            """
            - **Germany:** investigate the higher regional churn pattern.
            - **Inactive members:** prioritize activity and engagement campaigns.
            - **Older customers:** consider personalized retention communication.
            - **3+ products:** review product complexity and possible fee friction.
            - **High-balance customers:** prioritize carefully because losing them may
              have greater revenue impact.
            """
        )

    except Exception as e:
        st.error("Unable to load the dataset for the Data Insights page.")
        st.write(str(e))


# ===============================================================
# PAGE 3 — PREDICTION
# ===============================================================
else:

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter a customer's profile below to estimate their risk of churning "
        "so the retention team can prioritize outreach."
    )

    try:
        artifact = load_artifact()

        model = artifact["model"]
        scaler = artifact["scaler"]
        label_encoders = artifact["label_encoders"]
        feature_columns = artifact["feature_columns"]
        numeric_columns = artifact["numeric_columns"]
        categorical_columns = artifact["categorical_columns"]
        uses_scaled_input = artifact["uses_scaled_input"]
        target_mapping_inverse = artifact["target_mapping_inverse"]

        with st.form("churn_form"):
            col1, col2 = st.columns(2)

            with col1:
                credit_score = st.slider("Credit Score", 350, 850, 650)
                geography = st.selectbox(
                    "Geography",
                    label_encoders["Geography"].classes_
                )
                gender = st.selectbox(
                    "Gender",
                    label_encoders["Gender"].classes_
                )
                age = st.slider("Age", 18, 92, 38)
                tenure = st.slider("Tenure (years with bank)", 0, 10, 5)

            with col2:
                balance = st.number_input(
                    "Account Balance",
                    min_value=0.0,
                    max_value=260000.0,
                    value=75000.0,
                    step=1000.0
                )
                num_of_products = st.selectbox(
                    "Number of Products",
                    [1, 2, 3, 4]
                )
                has_cr_card = st.selectbox(
                    "Has Credit Card?",
                    ["Yes", "No"]
                )
                is_active_member = st.selectbox(
                    "Active Member?",
                    ["Yes", "No"]
                )
                estimated_salary = st.number_input(
                    "Estimated Salary",
                    min_value=0.0,
                    max_value=200000.0,
                    value=80000.0,
                    step=1000.0
                )

            submitted = st.form_submit_button("🔮 Predict Churn Risk")

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

            # Apply the same encoders used during training.
            for col in categorical_columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

            model_input = input_df.copy()

            if uses_scaled_input:
                model_input[numeric_columns] = scaler.transform(
                    input_df[numeric_columns]
                )

            prediction = model.predict(model_input)[0]
            probability = model.predict_proba(model_input)[0][1]
            label = target_mapping_inverse[prediction]

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if label == "Churned":
                    st.error(
                        f"⚠️ High churn risk — **{label}**"
                    )
                else:
                    st.success(
                        f"✅ Low churn risk — **{label}**"
                    )

            with result_col2:
                st.metric("Probability of Churn", f"{probability:.1%}")

            st.progress(min(max(float(probability), 0.0), 1.0))

            # ---------------------------------------------------
            # Business recommendation
            # ---------------------------------------------------
            st.subheader("💡 Recommended Business Action")

            reasons = []

            if is_active_member == "No":
                reasons.append(
                    "the customer is inactive, which is associated with higher churn"
                )

            if age >= 50:
                reasons.append(
                    "older customers show higher churn in this dataset"
                )

            if num_of_products >= 3:
                reasons.append(
                    "customers with 3+ products show higher churn"
                )

            if geography == "Germany":
                reasons.append(
                    "Germany has a noticeably higher churn rate"
                )

            if balance >= 100000:
                reasons.append(
                    "the customer has a high balance, increasing the potential "
                    "financial impact of losing the customer"
                )

            if probability >= 0.5:
                st.warning(
                    "Recommended action: **prioritize this customer for a "
                    "retention call or personalized offer.**"
                )

                if reasons:
                    st.write("Key risk factors:")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.write(
                        "Risk appears to be driven by a combination of customer attributes."
                    )

            else:
                st.info(
                    "Recommended action: **no urgent intervention is required.** "
                    "Continue standard engagement and monitor future activity."
                )

            st.caption(
                "Predictions are probabilistic estimates intended to support, "
                "not replace, retention-team judgment."
            )

    except FileNotFoundError:
        st.error(
            "model.pkl was not found. Make sure model.pkl is in the same "
            "GitHub folder as app.py."
        )
    except Exception as e:
        st.error("The prediction page could not load the model.")
        st.write(str(e))
