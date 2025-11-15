# Credit Card Fraud Detection

## 📌 Project Overview
This repository presents an **end-to-end machine learning solution** for detecting fraudulent credit card transactions. The project demonstrates applied skills in **data preprocessing, handling class imbalance, feature engineering, model development, evaluation, and deployment**. It leverages industry-standard tools and best practices, highlighting reproducibility and practical ML expertise.

---
## 📖 References
Kaggle Credit Card Fraud Dataset

Python Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit

## 🧰 Key Features
- **Models Implemented:**  
  - Logistic Regression  
  - Random Forest  
  - XGBoost 
  - Optional: Neural Networks  
- **Data Preprocessing:** Scaling, handling missing values, and encoding categorical variables  
- **Class Imbalance Handling:** SMOTE technique for robust training  
- **Serialized Components:**  
  - `credit_fraud_xgb.pkl` – Trained XGBoost model  
  - `features_means.pkl` – Precomputed feature means for consistent preprocessing  
- **Evaluation Metrics:** Precision, Recall, F1-score, AUC-ROC, and confusion matrix  
- **Optional Streamlit App:** Interactive dashboard for real-time fraud detection and model evaluation  

---



## ⚡ Quick Start

1. Install Dependencies
bash
pip install -r requirements.txt
2. Train Models
run the jupyter source file
3. Run Streamlit App (Optional)
bash
Copy code
streamlit run app/credit_streamlit.py

## 📊 Model Performance
Achieves high precision and recall, minimizing false negatives and false positives

Evaluated using AUC-ROC, F1-score, and confusion matrices

Optimized for imbalanced datasets, reflecting real-world transaction distributions

## 🎯 Objective
Develop a robust, reproducible, and deployable credit card fraud detection system while demonstrating:

Applied machine learning expertise

Data preprocessing and imbalance handling

Model evaluation and optimization

End-to-end pipeline development ready for production



💡 Skills Demonstrated
Applied ML algorithms in a real-world problem

Handling imbalanced datasets using advanced techniques

Model serialization and reproducibility

Interactive dashboard development with Streamlit

End-to-end data science workflow from preprocessing to deployment
