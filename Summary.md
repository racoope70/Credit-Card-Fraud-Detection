# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Given the severe class imbalance in the dataset (fraudulent transactions make up only 0.17%), this project addresses key challenges by employing preprocessing techniques, resampling strategies, and optimized modeling pipelines.

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
- **Class Distribution**:
  - Non-Fraudulent: 99.83%
  - Fraudulent: 0.17%

![Class Distribution](![image](https://github.com/user-attachments/assets/787d9e05-0983-4d8a-965a-493f4d9e8445)


---

## 2. Data Preprocessing <a name="data-preprocessing"></a>

### Key Steps:
1. **Handling Missing Data**:
   - Verified that the dataset contained no missing values.
2. **Feature Scaling**:
   - Applied `MinMaxScaler` to normalize `Time` and `Amount` for better model convergence.
3. **Handling Imbalanced Data**:

#### Why SMOTE?
The dataset is heavily imbalanced, as illustrated in the class distribution plot above. Without addressing this imbalance, machine learning models tend to be biased towards the majority class (non-fraudulent transactions), resulting in high accuracy but poor recall for the minority class.

**SMOTE (Synthetic Minority Oversampling Technique)** generates synthetic samples for the minority class by interpolating between existing samples, effectively balancing the dataset. This improves the model's ability to detect fraudulent transactions, reducing false negatives.

- **ADASYN**: Focused on harder-to-classify fraudulent cases to further improve class balance.

### Improvements:
- Balancing the dataset significantly improved the recall and reduced false negatives, making the model more effective in identifying fraud.

---

## 3. Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
### Key Findings:
1. **Class Imbalance**:
   - Fraudulent transactions constituted only 0.17% of the dataset.
2. **Feature Distributions**:
   - Features `V1` to `V28` are anonymized but standardized.
   - `Time` and `Amount` were skewed, requiring normalization.

### Visualizations:
#### Precision-Recall Curve:
![Precision-Recall Curve](https://github.com/user-attachments/assets/8e9c4bc6-7966-4f2f-b7e7-7d7c15f28f2d)

**Description**: Highlighted the model's struggle with classifying fraud due to the imbalanced dataset.

#### Confusion Matrix:
![Confusion Matrix](https://github.com/user-attachments/assets/90806960-734f-4700-9e98-aab45c591ad1)

**Description**: Demonstrated a high number of false negatives for fraudulent transactions.

#### ROC Curve:
![ROC Curve](https://github.com/user-attachments/assets/84ca4273-eb67-448f-81ef-ede526fafcc1)

**Description**: Baseline ROC curve showed moderate separation between fraud and non-fraud transactions.

---

## 4. Modeling <a name="modeling"></a>

### Models Tested:
1. **Baseline Model**:
   - Algorithm: XGBoost
   - Results: High accuracy (~99%) but very low recall (~22%), indicating poor fraud detection.
2. **Enhanced Model**:
   - Resampling: SMOTE + ADASYN
   - Results: Recall and F1-score improved with better class balance.
3. **Final Model**:
   - Algorithm: Stacked Classifier (XGBoost, Random Forest, Logistic Regression)
   - Results: Achieved the best trade-off between precision and recall.

### Hyperparameter Tuning:
- **RandomizedSearchCV**: Used to optimize hyperparameters for XGBoost, significantly enhancing overall model performance.

---

## 5. Evaluation <a name="evaluation"></a>

The table below summarizes the metrics at each iteration:

| Model                              | Precision | Recall | F1-Score | ROC-AUC |
|------------------------------------|-----------|--------|----------|---------|
| Baseline Model (XGBoost)           | 0.64      | 0.22   | 0.33     | 0.78    |
| After Resampling (SMOTE + ADASYN)  | 0.85      | 0.72   | 0.78     | 0.92    |
| Stacked Classifier + Optimization  | 0.91      | 0.88   | 0.89     | 0.98    |

### Confusion Matrix Analysis:
1. **Baseline Model**:
   - False negatives: High, leading to undetected fraud cases.
2. **Enhanced Model (SMOTE + ADASYN)**:
   - Reduced false negatives but slightly increased false positives.
3. **Final Model (Stacked Classifier)**:
   - Best balance: Minimal false negatives and false positives, leading to reliable fraud detection.

---

## 6. Future Directions <a name="future-directions"></a>

### Recommended Next Steps:
1. **Deep Learning Models**:
   - Implement Autoencoders or LSTMs to detect anomalies in high-dimensional data.
2. **Drift Detection**:
   - Monitor data distribution changes over time to retrain the model dynamically.
3. **Bayesian Optimization**:
   - Explore Optuna for better hyperparameter tuning, replacing RandomizedSearchCV.
4. **Comparative Study**:
   - Test additional anomaly detection techniques (e.g., Isolation Forest, One-Class SVM) and compare results.
5. **Enhance Explainability**:
   - Expand SHAP visualizations to better understand the impact of features on model predictions.

---

## References
1. [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
2. [SMOTE Paper](https://arxiv.org/abs/1106.1813)
3. [ADASYN Paper](https://ieeexplore.ieee.org/document/4633969)
