# Credit Card Fraud Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The primary challenge was handling a highly imbalanced dataset, where fraudulent transactions accounted for only 0.17% of the total. The project explored various approaches to preprocessing, modeling, and evaluating performance, emphasizing practical metrics like precision and recall.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Future Directions](#future-directions)

---

## 1. Dataset <a name="dataset"></a>
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: ~284,807 transactions
- **Features**:
  - 30 columns: `Time`, `Amount`, `Class` (target), and 27 anonymized features (`V1` to `V28`)
  - `Class`: 0 for non-fraudulent, 1 for fraudulent

---

## 2. Data Preprocessing <a name="data-preprocessing"></a>
### Key Steps:
1. **Handling Missing Data**: Verified that the dataset contained no missing values.
2. **Feature Scaling**: Applied `MinMaxScaler` to normalize `Time` and `Amount`.
3. **Handling Imbalanced Data**:
   - **SMOTE (Synthetic Minority Oversampling Technique)**: Generated synthetic samples to balance the minority class.
   - **ADASYN (Adaptive Synthetic Sampling)**: Focused on generating samples for harder-to-classify cases.

### Improvements:
- Balancing the dataset significantly improved recall and reduced false negatives in subsequent modeling stages.

---

## 3. Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
EDA was performed to understand the dataset's structure and identify patterns that could inform feature engineering and modeling.

### Findings:
1. **Class Imbalance**:
   - Fraudulent transactions: ~0.17%
   - Non-fraudulent transactions: ~99.83%
2. **Feature Distributions**:
   - Features `V1` to `V28` appeared standardized.
   - `Amount` and `Time` were skewed and required normalization.

### Visualizations:
#### Precision-Recall Curve:
![image](https://github.com/user-attachments/assets/8e9c4bc6-7966-4f2f-b7e7-7d7c15f28f2d)

**Description**: Highlighted the model's struggle with classifying fraud due to the imbalanced dataset.

#### Confusion Matrix:
![image](https://github.com/user-attachments/assets/90806960-734f-4700-9e98-aab45c591ad1)

**Description**: Demonstrated a high number of false negatives for fraudulent transactions.

#### ROC Curve:
![image](https://github.com/user-attachments/assets/84ca4273-eb67-448f-81ef-ede526fafcc1)

**Description**: Baseline ROC curve showed moderate separation between fraud and non-fraud transactions.

---

## 4. Modeling <a name="modeling"></a>
### Models Tested:
1. **Baseline Model**: XGBoost
   - **Performance**: Low recall due to class imbalance.
2. **Enhanced Model**: XGBoost + SMOTE + ADASYN
   - **Performance**: Improved recall and F1-score.
3. **Final Model**: Stacked Classifier (XGBoost, Random Forest, Logistic Regression)
   - **Performance**: Achieved the best balance between precision and recall.

### Hyperparameter Tuning:
- **RandomizedSearchCV** was used to optimize hyperparameters for XGBoost, significantly enhancing performance.

---

## 5. Evaluation <a name="evaluation"></a>
The table below summarizes how key metrics improved across iterations:

| Model                              | Precision | Recall | F1-Score | ROC-AUC |
|------------------------------------|-----------|--------|----------|---------|
| Baseline Model (XGBoost)           | 0.64      | 0.22   | 0.33     | 0.78    |
| After Resampling (SMOTE + ADASYN)  | 0.85      | 0.72   | 0.78     | 0.92    |
| Stacked Classifier + Optimization  | 0.91      | 0.88   | 0.89     | 0.98    |

### Key Observations:
1. **Recall**: Improved from 0.22 (Baseline) to 0.88 (Final Model), drastically reducing false negatives.
2. **Precision**: Increased to 0.91, ensuring high trust in fraud detection.
3. **F1-Score**: Improved steadily, reflecting a balanced trade-off between precision and recall.

---

## 6. Future Directions <a name="future-directions"></a>
### Next Steps:
1. **Experiment with Deep Learning**:
   - Use Autoencoders or LSTMs for anomaly detection in high-dimensional data.
2. **Implement Drift Detection**:
   - Monitor data changes over time to retrain the model as needed.
3. **Optimize Further**:
   - Use Bayesian Optimization (e.g., Optuna) to refine hyperparameters further.
4. **Enhance Feature Engineering**:
   - Explore temporal patterns and additional external features that could improve prediction.

