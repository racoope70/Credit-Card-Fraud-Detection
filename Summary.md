# Credit Card Fraud Detection Project

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Access and Overview](#dataset-access-and-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building and Iterations](#model-building-and-iterations)
    - [Baseline Model](#baseline-model)
    - [SMOTE + ADASYN](#smote--adasyn)
    - [Stacking and Hyperparameter Tuning](#stacking-and-hyperparameter-tuning)
6. [Model Evaluation](#model-evaluation)
7. [Feature Importance and Explainability](#feature-importance-and-explainability)
8. [Deployment and Future Improvements](#deployment-and-future-improvements)

---

## 1. Introduction <a name="introduction"></a>
- **Objective**: To detect fraudulent transactions in a credit card dataset.
- **Goal**: Build a model with high precision and recall while minimizing false positives.
- **Dataset**: Credit Card Fraud Detection Dataset from Kaggle.

---

## 2. Dataset Access and Overview <a name="dataset-access-and-overview"></a>
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- **Description**:
  - Transactions: ~284,807.
  - Features: 30 columns (anonymized numerical features: V1–V28, Time, Amount).
  - Target Variable: `Class` (0 for non-fraudulent, 1 for fraudulent).

---

## 3. Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
### **Steps**
1. Check class distribution.
2. Visualize distributions of features (e.g., histograms, boxplots).
3. Analyze relationships between features and the target variable.

### **Code Example**
```python
# Visualizing the class distribution
sns.countplot(data=df, x='Class')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Visualizing a numerical feature (Amount)
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()
