#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and saved mean feature values
xgb_model = joblib.load("credit_fraud_xgb.pkl")
feature_means = joblib.load("feature_means.pkl")  # a dict or pd.Series

explainer = shap.TreeExplainer(xgb_model)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# -----------------
# 1️⃣ Top Features Only for User Input.
# -----------------
top_features = ['V14', 'V17', 'Amount_scaled', 'V12', 'V10']
user_input = {}

st.header("Transaction Input")
for feature in top_features:
    default = float(feature_means[feature])
    if 'Amount' in feature:
        user_input[feature] = st.slider(f"{feature} (scaled)", 0.0, 1.0, default)
    else:
        user_input[feature] = st.number_input(f"{feature}", value=default)

# -----------------
# 2️⃣ Automatically Fill Remaining Features
# -----------------
for feature in xgb_model.feature_names_in_:
    if feature not in user_input:
        user_input[feature] = feature_means[feature]

# -----------------
# 3️⃣ Create DataFrame in Correct Order
# -----------------
input_df = pd.DataFrame([user_input])
input_df = input_df[xgb_model.feature_names_in_]

# -----------------
# 4️⃣ Prediction
# -----------------
fraud_prob = xgb_model.predict_proba(input_df)[0][1]
st.subheader(f"Fraud Probability: {fraud_prob:.2%}")

# -----------------
# 5️⃣ SHAP Explanation
# -----------------
shap_values = explainer.shap_values(input_df)
shap_df = pd.DataFrame({
    'Feature': input_df.columns,
    'SHAP Value': shap_values[0]
}).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)

st.subheader("Top Feature Contributions (SHAP)")
st.dataframe(shap_df)

# Horizontal bar chart
colors = ['red' if val > 0 else 'green' for val in shap_df['SHAP Value']]
plt.figure(figsize=(8,5))
sns.barplot(x='SHAP Value', y='Feature', data=shap_df, palette=colors)
plt.title("Top Feature Contributions")
plt.xlabel("SHAP Value (Impact on Fraud Probability)")
plt.ylabel("Feature")
plt.tight_layout()
st.pyplot(plt)


# In[ ]:




