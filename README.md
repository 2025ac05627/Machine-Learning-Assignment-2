# Bank Term Deposit Classification - ML Assignment 2

## a. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification algorithms to predict whether a banking client will subscribe to a term deposit (binary response variable: 'yes' or 'no'). This allows financial institutions to run targeted direct marketing campaigns efficiently.

## b. Dataset Description
- **Dataset Name:** UCI Bank Marketing Dataset
- **Number of Instances:** 45,211 rows
- **Number of Features:** 16 features (including age, job, marital status, education, default, balance, housing, loan, contact duration, and campaign history)
- **Target Variable:** `target` (Binary: 1 = Subscribed, 0 = Not Subscribed)

## c. GitHub Repository Link
[https://github.com/2025ac05627/Machine-Learning-Assignment-2/tree/main](https://github.com/2025ac05627/Machine-Learning-Assignment-2/tree/main)

## d. Models Used & Performance Evaluation

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8914 | 0.8726 | 0.5945 | 0.2259 | 0.3274 | 0.3205 |
| **Decision Tree** | 0.8769 | 0.7044 | 0.4743 | 0.4792 | 0.4767 | 0.4070 |
| **kNN** | 0.8923 | 0.8089 | 0.5717 | 0.3166 | 0.4075 | 0.3724 |
| **Naive Bayes** | 0.8445 | 0.8160 | 0.3659 | 0.4490 | 0.4032 | 0.3171 |
| **Random Forest (Ensemble)** | **0.9073** | **0.9276** | **0.6622** | **0.4244** | **0.5173** | **0.4830** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers strong baseline accuracy (89.14%) and AUC (0.8726), but struggles with class imbalance resulting in a very low Recall (22.59%). |
| **Decision Tree** | Maintains a solid balance between Precision (47.43%) and Recall (47.92%), but drops in overall AUC (0.7044) due to sensitivity to feature splits. |
| **kNN** | Yields high accuracy (89.23%) and decent AUC (0.8089) with normalized features, but misses many positive instances leading to low Recall (31.66%). |
| **Naive Bayes** | Achieves a higher Recall (44.90%) than linear models by leveraging feature probabilities, but trade-offs result in lower Precision (36.59%) and Accuracy (84.45%). |
| **Random Forest (Ensemble)** | Outperforms all individual classifiers across every evaluation metric (Accuracy: 90.73%, AUC: 0.9276, F1: 0.5173, MCC: 0.4830) by reducing variance and handling class complexity effectively. |
| **Overall Winner** | **Random Forest (Ensemble)** is the overall winner for this dataset. |
