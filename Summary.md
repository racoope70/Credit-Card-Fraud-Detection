# Credit Card Fraud Detection Project

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Access and Overview](#dataset-access-and-overview)
3. [Project Structure](#project-structure)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Graph 1: Precision-Recall Curve](#graph-1-precision-recall-curve)
   - [Graph 2: Confusion Matrix](#graph-2-confusion-matrix)
   - [Graph 3: ROC Curve](#graph-3-roc-curve)
5. [Data Preprocessing](#data-preprocessing)
   - [Graph 4: Resampling Precision-Recall Curve](#graph-4-resampling-precision-recall-curve)
   - [Graph 5: Resampling Confusion Matrix](#graph-5-resampling-confusion-matrix)
   - [Graph 6: Resampling ROC Curve](#graph-6-resampling-roc-curve)
6. [Model Building and Iterations](#model-building-and-iterations)
   - [Graph 7: Stacked Classifier Precision-Recall Curve](#graph-7-stacked-classifier-precision-recall-curve)
   - [Graph 8: Stacked Classifier Confusion Matrix](#graph-8-stacked-classifier-confusion-matrix)
   - [Graph 9: Stacked Classifier ROC Curve](#graph-9-stacked-classifier-roc-curve)
7. [Model Evaluation](#model-evaluation)
8. [Feature Importance and Explainability](#feature-importance-and-explainability)
9. [Deployment and Future Improvements](#deployment-and-future-improvements)

---

## 1. Introduction <a name="introduction"></a>
Credit card fraud detection is a critical problem for financial institutions due to its impact on customer trust and operational costs. This project leverages machine learning techniques to identify fraudulent transactions while maintaining a balance between precision and recall.

The initial dataset showed significant class imbalance, with only ~0.17% fraudulent transactions. Addressing this imbalance was a key challenge to ensure accurate predictions for both fraud and non-fraud cases.

---

## 2. Dataset Access and Overview <a name="dataset-access-and-overview"></a>
- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets).
- **Description**:
  - **Transactions**: ~284,807.
  - **Features**: 30 columns (anonymized numerical features: V1–V28, Time, Amount).
  - **Target Variable**: Class (0 for non-fraudulent, 1 for fraudulent).

---

## 3. Project Structure <a name="project-structure"></a>
The project is organized to facilitate reproducibility and modularity:

   pip install -r requirements.txt
Make sure you have Python 3.7+ installed.

## Project Structure
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   └── output/                  # Model predictions and outputs
│
├── notebooks/
│   ├── 1_EDA.ipynb              # Exploratory Data Analysis
│   ├── 2_Preprocessing.ipynb    # Data Preprocessing and Feature Engineering
│   ├── 3_Model_Building.ipynb   # Model Building and Evaluation
│
├── src/
│   ├── data_preprocessing.py    # Data preprocessing scripts
│   ├── model_training.py        # Model training and optimization
│   ├── evaluation.py            # Model evaluation and metrics
│   └── utils.py                 # Utility functions
│
├── reports/
│   ├── figures/                 # Graphs and plots
│   └── final_report.md          # Final report
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── config.yaml                  # Configuration settings


## 4. Exploratory Data Analysis (EDA) <a name="exploratory-data-analysis-eda"></a>
Exploratory Data Analysis (EDA) was performed to gain initial insights into the dataset and address key challenges, such as class imbalance. The analysis highlighted the imbalance between fraudulent and non-fraudulent transactions, which required careful consideration during modeling.


### Graph 1: Confusion Matrix <a name="graph-2-confusion-matrix"></a>
![Confusion Matrix](![image](https://github.com/user-attachments/assets/d1a00339-ef4a-4b41-88c3-cfef8497d5b5)

- **Description**: The initial precision-recall curve revealed the model's struggles with classifying fraud due to class imbalance.

### Graph 2: Precision-Recall Curve <a name="graph-1-precision-recall-curve"></a>
![Precision-Recall Curve](![image](https://github.com/user-attachments/assets/46026b2f-0121-4f38-ac03-b1fb69599953)


- **Description**: The confusion matrix demonstrates a high number of false negatives for fraudulent transactions.

### Graph 3: ROC Curve <a name="graph-3-roc-curve"></a>
![ROC Curve](![image](https://github.com/user-attachments/assets/e7791f04-20ac-448d-bdde-48d61bb8fc4b)


- **Description**: The baseline ROC curve highlights moderate separation between fraud and non-fraud transactions.

---

## 5. Data Preprocessing <a name="data-preprocessing"></a>
To address class imbalance and enhance model performance, data preprocessing steps included feature scaling and the application of resampling techniques like SMOTE and ADASYN.

### Key Steps:
1. **Feature Scaling**: MinMaxScaler was applied to normalize numerical features (`Time` and `Amount`).
2. **Resampling**: Synthetic samples for the minority class (fraudulent transactions) were generated using SMOTE and ADASYN to improve balance.

### Graph 4: Resampling Confusion Matrix <a name="graph-5-resampling-confusion-matrix"></a>
![Resampling Confusion Matrix](![image](https://github.com/user-attachments/assets/a6969cef-c34c-492b-9871-d421079c1bd4)


- **Description**: The confusion matrix post-resampling shows improved detection of fraud cases.

### Graph 5: Resampling Precision-Recall Curve <a name="graph-4-resampling-precision-recall-curve"></a>
![Resampling Precision-Recall Curve](![image](https://github.com/user-attachments/assets/d040797e-bf5a-41b4-91de-108b1a041522)


- **Description**: Precision-recall curve after resampling shows improved recall for fraudulent transactions while maintaining precision.

### Graph 6: Resampling ROC Curve <a name="graph-6-resampling-roc-curve"></a>
![Resampling ROC Curve](![image](https://github.com/user-attachments/assets/a13ae8e0-ee9a-4a60-aa53-fc51dc5caf8c)


- **Description**: The ROC curve demonstrates significant improvement in model discrimination after resampling.

---

## 6. Model Building and Iterations <a name="model-building-and-iterations"></a>
Model building involved experimenting with different algorithms and combining them into a stacked classifier. Hyperparameter tuning further enhanced performance.

### Key Enhancements:
1. **Baseline Model**: XGBoost with default settings showed limitations in handling the class imbalance.
2. **Stacked Classifier**: Combined the strengths of XGBoost, Random Forest, and Logistic Regression to boost performance.
3. **Hyperparameter Tuning**: Randomized search was applied to optimize XGBoost parameters.


### Graph 7: Stacked Classifier Confusion Matrix <a name="graph-8-stacked-classifier-confusion-matrix"></a>
![Stacked Classifier Confusion Matrix](![image](https://github.com/user-attachments/assets/9ed68f2c-0119-4f59-a6d8-1b6a7670bd71)


- **Description**: Confusion matrix for the stacked classifier shows a significant reduction in false negatives.

### Graph 8: Stacked Classifier Precision-Recall Curve <a name="graph-7-stacked-classifier-precision-recall-curve"></a>
![Stacked Classifier Precision-Recall Curve](![image](https://github.com/user-attachments/assets/91a135ac-5829-460a-aab2-a7a5f698cf5d)


- **Description**: Precision-recall curve for the stacked classifier highlights improved recall and precision balance.

### Graph 9: Stacked Classifier ROC Curve <a name="graph-9-stacked-classifier-roc-curve"></a>
![Stacked Classifier ROC Curve](![image](https://github.com/user-attachments/assets/926f4514-47dc-459e-a27b-2b434cb49114)


- **Description**: ROC curve for the stacked classifier demonstrates near-optimal performance.

---


## 7. Model Evaluation <a name="model-evaluation"></a>
Model performance was evaluated using key metrics across each iteration of the pipeline. Below is an overview of how the numbers improved at each step.

### Baseline Model:
- **Precision**: 0.64
- **Recall**: 0.22
- **F1-Score**: 0.33
- **ROC-AUC Score**: 0.78
- **Observation**: The baseline XGBoost model struggled to detect fraudulent transactions due to severe class imbalance, leading to low recall and F1-score.

### After Resampling (SMOTE + ADASYN):
- **Precision**: 0.85
- **Recall**: 0.72
- **F1-Score**: 0.78
- **ROC-AUC Score**: 0.92
- **Observation**: The application of resampling techniques significantly boosted recall and F1-score by balancing the class distribution, enabling the model to identify more fraudulent transactions.

### Stacked Classifier with Hyperparameter Tuning:
- **Precision**: 0.91
- **Recall**: 0.88
- **F1-Score**: 0.89
- **ROC-AUC Score**: 0.98
- **Observation**: The introduction of a stacking classifier (XGBoost, Random Forest, Logistic Regression) and hyperparameter optimization maximized both precision and recall. The ROC-AUC score approached 0.98, demonstrating excellent discrimination between fraud and non-fraud classes.

### Comparison Table:
| Iteration                          | Precision | Recall | F1-Score | ROC-AUC |
|------------------------------------|-----------|--------|----------|---------|
| Baseline Model                     | 0.64      | 0.22   | 0.33     | 0.78    |
| After Resampling (SMOTE + ADASYN)  | 0.85      | 0.72   | 0.78     | 0.92    |
| Stacked Classifier + Optimization  | 0.91      | 0.88   | 0.89     | 0.98    |

### Observations:
1. **Recall** improved the most across iterations, rising from 0.22 (Baseline) to 0.88 (Final Model). This indicates the model became significantly better at identifying fraud cases.
2. **F1-Score** steadily increased with better class balance and model complexity, indicating an optimal trade-off between precision and recall.
3. **Precision** rose to 0.91, minimizing false positives and ensuring a higher trust level in detected fraud cases.
4. **ROC-AUC Score** improved from 0.78 to 0.98, reflecting the model's enhanced ability to discriminate between fraud and non-fraud classes.

---

This section provides a detailed view of the model's journey from baseline performance to optimized results. Let me know if you need further adjustments or details!```

---

## 8. Feature Importance and Explainability <a name="feature-importance-and-explainability"></a>
SHAP (SHapley Additive exPlanations) visualizations were used to interpret model predictions. The most influential features for detecting fraud were identified, improving trust and transparency in the model.

---

## 9. Deployment and Future Improvements <a name="deployment-and-future-improvements"></a>

### Deployment:
The final model was saved using `pickle` and deployed as an API with FastAPI for real-time predictions.

### Future Improvements:
1. **Deep Learning Models**: Experiment with Autoencoders or LSTMs to detect anomalies in high-dimensional data.
2. **Drift Detection**: Implement a drift detection mechanism to monitor and retrain the model as data evolves.
3. **Bayesian Optimization**: Fine-tune hyperparameters using techniques like Optuna for even better performance.
