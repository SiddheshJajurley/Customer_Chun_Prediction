# Customer Churn Prediction
## Overview
This project aims to predict customer churn using machine learning. The dataset is the Telco Customer Churn dataset, which includes information about a telecom company’s customers, such as demographics, account information, and service usage.

The project walks through data cleaning, preprocessing, class imbalance handling, model training, and evaluation using a robust and modular pipeline.

## Workflow

### 1️⃣ Data Exploration
- Loaded the Telco dataset and explored:
  - Missing values (TotalCharges column).
  - Categorical vs numerical features.
  - Class distribution (churn rate).

### 2️⃣ Data Cleaning & Preprocessing
- Converted TotalCharges to numeric and handled missing values.
- Dropped irrelevant columns (customerID).
- Split data into features (X) and target (y).
- Applied:
  - StandardScaler for numeric columns.
  - OneHotEncoder (drop-first, handle-unknown) for categorical columns.

### 3️⃣ Handling Class Imbalance
- The dataset was imbalanced (fewer churn cases). To address this:
  - Applied SMOTE + ENN (SMOTEENN) to oversample the minority class and clean the data.

### 4️⃣ Modeling
- **Train-Test Split**: 80% training / 20% testing, stratified.
- **Model Used**: RandomForestClassifier (baseline).

### 5️⃣ Evaluation
The model achieved strong performance on the test set:

| Metric      | Score   |
|-------------|---------|
| F1-score    | 0.9604  |
| AUC-ROC     | 0.9876  |
| Accuracy    | 0.95    |

#### Detailed Classification Report:

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| 0 (Not Churn)    | 0.95      | 0.94   | 0.95     | 557     |
| 1 (Churn)        | 0.96      | 0.96   | 0.96     | 744     |

## Technologies & Libraries
- **Python**
- **Pandas, Numpy, Matplotlib**
- **Scikit-learn**:
  - Pipelines, preprocessing, feature selection
  - RandomForestClassifier, LogisticRegression
- **Imbalanced-learn**:
  - SMOTEENN for handling imbalanced data



## Improvements & Next Steps

- **Hyperparameter tuning**: Use `GridSearchCV`, `RandomizedSearchCV`, or `Bayesian Optimization`.
- **Model comparisons**: Try `XGBoost`, `LightGBM`, or other models for improved performance.
- **Feature importance visualization and SHAP analysis**: Understand the key features driving churn prediction.
- **Deployment**: Wrap the model as a web service or dashboard (e.g., using Flask, Streamlit).

## License
This project is free to use for educational and practical purposes. 🚀

---

## Additional Suggestions

### 1️⃣ Data Exploration
- **Correlation Matrix**: Plot a heatmap of correlations to identify relationships between features.
- **Missing Data Analysis**: Visualize missing data patterns using `missingno` for a clearer understanding.

### 2️⃣ Data Cleaning & Preprocessing
- **Outlier Detection**: Before scaling, check for outliers (especially in numerical columns) and decide whether to handle them (e.g., using IQR or Z-scores).
- **Feature Engineering**: Create new features, such as tenure length or monthly charge trends, based on domain knowledge.

### 3️⃣ Handling Class Imbalance
- **Other Resampling Techniques**: Experiment with other techniques like ADASYN or RandomUnderSampling.
- **Cost-sensitive Learning**: Adjust the model’s loss function to be more sensitive to the minority class (e.g., adjusting class weights in `RandomForestClassifier`).

### 4️⃣ Modeling
- **Model Comparison**: While `RandomForestClassifier` is a great baseline, test other models like `XGBoost`, `Logistic Regression`, and `LightGBM`.
- **Ensemble Methods**: Explore stacking or boosting approaches to combine multiple models' strengths.

### 5️⃣ Evaluation
- **Cross-validation**: Use cross-validation (e.g., StratifiedKFold) for a more robust evaluation of model performance.
- **Confusion Matrix & ROC Curves**: Visualize the confusion matrix to better understand model classification performance. Plot ROC curves for different models to compare visually.

### 6️⃣ Improvements & Next Steps
- **Hyperparameter Tuning**: Implement `RandomizedSearchCV` or `Bayesian Optimization` for hyperparameter tuning.
- **Model Interpretability**: Use `LIME` (Local Interpretable Model-agnostic Explanations) for model-specific predictions.
- **Model Deployment**: For deployment, consider `Flask`/`Django` for creating a REST API or `Streamlit` for creating an interactive web dashboard.

### 7️⃣ Documentation & Future
- **Model Drift**: Plan for model monitoring and retraining as new customer data is collected to avoid performance decay over time.
- **Collaborative Filtering**: If the dataset includes interaction data (e.g., customer-agent interactions), collaborative filtering can help enhance predictions.


