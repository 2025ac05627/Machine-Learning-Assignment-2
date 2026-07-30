import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Page Config
st.set_page_config(page_title="ML Classifier Evaluation | Vinod Ranganathan", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .sidebar-author {
        text-align: center;
        padding: 12px;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 40px;
        border-top: 1px solid #E2E8F0;
        padding-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Sidebar - Author & Controls
st.sidebar.markdown("""
    <div class="sidebar-author">
        <h3 style="margin:0; color:#F8FAFC;">👨‍💻 Vinod Ranganathan</h3>
        <small style="color:#94A3B8;">ML Assignment 2 Presenter</small>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (test data)", type=["csv"])

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
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl"
}

# Load Scaler if available
scaler = joblib.load("model/scaler.pkl") if os.path.exists("model/scaler.pkl") else None

st.title("📊 Machine Learning Classification Web App")
st.markdown("Upload test data, evaluate models in real-time, and compare performance across algorithms.")

# Helper function to prepare input features & targets safely
def prepare_data(df_input, model, scaler=None):
    data = df_input.copy()
    
    # Drop unique ID if present
    if 'customerID' in data.columns:
        data.drop(columns=['customerID'], inplace=True)
        
    # Clean TotalCharges if present
    if 'TotalCharges' in data.columns and data['TotalCharges'].dtype == 'object':
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(" ", np.nan), errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Extract target column
    target_col = 'target' if 'target' in data.columns else ('Churn' if 'Churn' in data.columns else None)
    if not target_col:
        return None, None
        
    X = data.drop(columns=[target_col])
    raw_y = data[target_col]
    
    # Map target strings to binary 0/1
    if raw_y.dtype == 'object':
        y = raw_y.map({'No': 0, 'Yes': 1, 'no': 0, 'yes': 1})
    else:
        y = raw_y.astype(int)

    # One-hot encoding for categorical features if raw features uploaded
    if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
        X = pd.get_dummies(X, drop_first=True)

    # Reindex columns to match model/scaler training feature names and order
    expected_cols = None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        expected_cols = scaler.feature_names_in_
    elif hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_

    if expected_cols is not None:
        X = X.reindex(columns=expected_cols, fill_value=0)
        
    return X, y

# Tabs Definition
tab1, tab2 = st.tabs(["📊 Single Model Evaluation", "🏆 All Models Comparison"])

# TAB 1: SINGLE MODEL EVALUATION
with tab1:
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        col_prev, col_stat = st.columns([2, 1])
        with col_prev:
            st.subheader("📋 Dataset Preview")
            st.dataframe(data.head(), use_container_width=True)
        with col_stat:
            st.subheader("ℹ️ Dataset Summary")
            st.write(f"**Total Rows:** {data.shape[0]:,}")
            st.write(f"**Total Columns:** {data.shape[1]}")

        try:
            model = joblib.load(model_file_map[model_option])
            X_test, y_test = prepare_data(data, model, scaler=scaler)

            if X_test is None or y_test is None:
                st.warning("Please ensure your uploaded CSV contains a `'target'` or `'Churn'` column.")
            else:
                # Apply feature scaling for Logistic Regression / kNN
                if model_option in ["Logistic Regression", "kNN"] and scaler is not None:
                    try:
                        X_eval = scaler.transform(X_test)
                    except Exception:
                        X_eval = X_test
                else:
                    X_eval = X_test

                # Generate Predictions
                y_pred = model.predict(X_eval)
                y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

                # Metric Calculations
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)

                # Performance KPI Cards
                st.subheader(f"📈 Performance Metrics: {model_option}")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

                st.markdown("---")

                # Visualizations
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("🧩 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['Class 0', 'Class 1'],
                                yticklabels=['Class 0', 'Class 1'])
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    st.pyplot(fig)

                with col_right:
                    st.subheader("📝 Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0, color="#D1E7DD"), use_container_width=True)

        except Exception as e:
            st.error(f"Error evaluating model `{model_option}`: {e}")
    else:
        st.info("👈 Please upload a CSV test dataset from the sidebar to begin.")

# TAB 2: DYNAMIC ALL MODELS COMPARISON
with tab2:
    st.subheader("🏆 Dynamic Multi-Model Comparison Leaderboard")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        results = []

        for name, path in model_file_map.items():
            try:
                model = joblib.load(path)
                X_test, y_test = prepare_data(data, model, scaler=scaler)

                if X_test is not None and y_test is not None:
                    if name in ["Logistic Regression", "kNN"] and scaler is not None:
                        try:
                            X_eval = scaler.transform(X_test)
                        except Exception:
                            X_eval = X_test
                    else:
                        X_eval = X_test

                    y_pred = model.predict(X_eval)
                    y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred

                    results.append({
                        "ML Model Name": name,
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "AUC": round(roc_auc_score(y_test, y_proba), 4),
                        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
                        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
                    })
            except Exception as e:
                st.warning(f"Could not evaluate `{name}`: {e}")

        if results:
            summary_df = pd.DataFrame(results)
            st.dataframe(
                summary_df.style.highlight_max(subset=["Accuracy", "AUC", "F1 Score", "MCC"], color="#C6F6D5"),
                use_container_width=True
            )
        else:
            st.error("Unable to calculate metrics. Check dataset target columns.")
    else:
        st.info("Upload `test_data.csv` in the sidebar to automatically calculate metrics across all 5 models.")

# Footer
st.markdown("""
    <div class="footer">
        ML Classifier Evaluation Dashboard | Presented by Vinod Ranganathan
    </div>
""", unsafe_allow_html=True)
