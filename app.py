import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation", layout="wide")

st.title("📊 Machine Learning Classification Web App")
st.markdown("Upload test data, pick a model, and evaluate performance metrics in real time.")

# 1. Dataset Upload Option (CSV)
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (test data)", type=["csv"])

# 2. Model Selection Dropdown
st.sidebar.header("2. Choose Model")
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest (Ensemble)"]
)

model_file_map = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_(ensemble).pkl"
}

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head())

    if 'target' in data.columns:
        X_test = data.drop(columns=['target'])
        y_test = data['target']

        # Load selected model
        try:
            model = joblib.load(model_file_map[model_option])
            
            # Prediction
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = y_pred

            # Calculate Metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            # 3. Display Evaluation Metrics
            st.subheader(f"📈 Performance Metrics: {model_option}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")

            # 4. Confusion Matrix & Classification Report
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                st.pyplot(fig)

            with col_right:
                st.subheader("📝 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

        except Exception as e:
            st.error(f"Error loading model or generating predictions: {e}")
    else:
        st.warning("Please ensure your uploaded CSV has a 'target' column for ground truth labels.")
else:
    st.info("👈 Please upload a CSV test dataset from the sidebar to begin.")
