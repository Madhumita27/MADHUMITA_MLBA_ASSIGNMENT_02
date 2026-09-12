# Customer Churn Prediction — Business Decision-Support App

An end-to-end machine learning project that predicts which bank customers are
at risk of churning (closing their account), so the retention team can
prioritize outreach.

**Project flow:** Business Problem → Dataset → Data Preparation → EDA → ML
Model → Evaluation → Prediction → Business Recommendation → Streamlit App →
GitHub → Deployment

---

## 1. Business Problem

A retail bank is losing customers at a meaningful rate, which directly erodes
recurring revenue and increases the cost of acquiring replacement customers.

**Objective:** predict, from a customer's profile and account activity, the
probability that they will churn — *before* they leave — so the retention
team can proactively intervene with the highest-risk customers under a
limited outreach budget.

**Target variable:** `Exited` (1 = churned, 0 = retained).

## 2. Dataset

**Bank customer churn dataset** (`data/dataset.csv`) — 10,000 customers,
11 columns, covering demographics (age, gender, geography), account details
(credit score, balance, tenure, number of products), and engagement
(active member status, credit card ownership).

> **Honest note on this data:** `data/dataset.csv` in this repo is an
> **illustrative / example dataset**, built to match the column names,
> value ranges, and known churn patterns (e.g. higher churn in Germany, among
> inactive and older customers) of the well-known Kaggle "Churn Modelling"
> dataset. **It is not a literal download of the original Kaggle file.**
> If your assignment requires a dataset sourced from an actual public
> repository (Kaggle/UCI/etc.), download the real `Churn_Modelling.csv` from
> Kaggle and drop it in at `data/dataset.csv` with the same column names —
> the notebook and app run unchanged, since they only depend on those column
> names. Do not describe this example file as the original Kaggle download
> in your submission.

## 3. Data Preparation

- Verified there are no missing values and no exact duplicate rows in this
  export (checked programmatically, not assumed).
- Drops `RowNumber`, `CustomerId`, `Surname` if present — identifiers with
  no predictive value (the current example file doesn't include them, but
  the cleaning step handles a raw export that does).
- Encoded `Geography` and `Gender` with `LabelEncoder`.
- Standardized numeric features with `StandardScaler` (fit on the training
  set only, to avoid data leakage).

## 4. Exploratory Data Analysis

Key findings (see `notebooks/model_development.ipynb` for full charts):

- Overall churn rate is ~20% (moderately imbalanced classes).
- **Germany** has a noticeably higher churn rate than France or Spain.
- Churned customers skew **older** than retained customers.
- Churned customers carry **higher average balances** — losing them has an
  outsized revenue impact.
- **Inactive members** churn at roughly double the rate of active members.
- Customers holding **3+ products** churn more than those with 1–2,
  suggesting product complexity or fee friction rather than loyalty.

## 5. Machine Learning Model

Two models were trained and compared on the same 80/20 train/test split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7015 | 0.3770 | 0.7778 | 0.5078 | 0.7833 |
| Random Forest | 0.7395 | 0.4043 | 0.6667 | 0.5033 | 0.7921 |

**Model selected: Logistic Regression.** For a retention use case, missing an
actual churner (false negative) is costlier than a wasted retention offer to
a loyal customer (false positive) — so **recall on the churn class** was
prioritized over raw accuracy. Logistic Regression achieved the higher
recall of the two models tested (0.78 vs 0.67), despite Random Forest's
slightly higher accuracy and ROC-AUC. It's also fully interpretable, which
helps when explaining a prediction to a retention manager.

## 6. Business Recommendations

1. Prioritize outreach to **inactive, older, higher-balance** customers —
   highest predicted churn risk *and* highest revenue impact if lost.
2. Investigate the **German market** specifically for a structural,
   region-specific retention issue.
3. Review the **3+ products customer experience** — more products correlates
   with more churn, not less.
4. Run an **activity-driving campaign** — engagement is one of the strongest,
   most actionable levers available.
5. Use the model's **churn probability score** to rank customers for a
   monthly retention campaign under a fixed budget, rather than treating all
   customers equally.

## 7. Repository Structure

```
business-analytics-project/
├── app.py                              # Streamlit prediction app
├── model.pkl                           # trained model + scaler + encoders (bundled)
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv                     # dataset used for modeling (see note in Section 2)
├── notebooks/
│   └── model_development.ipynb         # full EDA + modeling notebook
└── images/
    └── dashboard.png                   # add a screenshot of your deployed app here
```

## 8. Running the App Locally

```bash
git clone <your-repo-url>
cd business-analytics-project
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 9. Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (including `model.pkl` — it's small enough
   to commit directly).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, select this repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a public URL.
5. Take a screenshot of the running app and save it as `images/dashboard.png`
   in the repo.
6. Add the live URL and your GitHub repo link below (and in your
   assignment submission).

**GitHub Repository:** `<add your repo URL here>`
**Live Application:** `<add your Streamlit Cloud URL here>`

## 10. How to Use the App

1. Enter a customer's credit score, geography, gender, age, tenure, balance,
   number of products, credit card ownership, and active-member status.
2. Click **Predict Churn Risk**.
3. The app shows the predicted outcome (Churned / Retained), the churn
   probability, and a tailored business recommendation based on the
   customer's specific risk factors.
