# Patient-Acceptance-Prediction
This project develops machine learning models to predict whether patients referred to the Agency for Integrated Care (AIC) are likely to accept community care services. The models identify key factors influencing patient acceptance and provide decision support for referral planning and healthcare resource allocation.

## Problem Statement
To develop a predictive model using referral data to analyze patterns, identify key factors and generate actionable insights for effective decision-making

## Objectives
- Which factors determine whether patients are more likely to accept or reject services
- Whether access to community-based services delay the need for residential care
- Which factors in patients that utilize community care delay the need for residential care

## Data Flow
1. `leakage_check.py`
   - Detects potential data leakage by ensuring no target information is unintentionally included in the training features.
2. `filtering.py`
   - Cleans the referral dataset by removing invalid records, handling missing values, and selecting relevant patient information.
3. `filtering_test.py`
   - Verifies that the filtering process preserves data integrity and produces a clean dataset for model training.
4. Model Training
   - Trains and evaluates multiple machine learning models using the processed dataset.
5. Model Evaluation
   - Compares model performance using accuracy, F1-score, and other evaluation metrics.

## Results
- Ensemble Model Accuracy: **66.9%**
- Temporal Prediction Model Accuracy: **69.26%**
- Identified the most influential factors affecting patient acceptance.
- Developed a scalable decision-support tool for referral planning.

## Solution
A predictive analytics pipeline was developed to help the Agency for Integrated Care (AIC) identify patients who are more likely to accept community care services and provide actionable insights for referral planning.

The solution consists of the following stages:

1. **Data Cleaning**
   - Removed inconsistent and noisy referral records.
   - Standardized categorical variables.
   - Merged multiple datasets using the application ID to create a unified dataset.

2. **Feature Selection**
   - Ranked the most important predictive variables using Mutual Information, Cramér's V, and Random Forest feature importance.
   - Selected the most informative features for model training.

3. **Model Development**
   - Evaluated multiple machine learning models, including:
     - XGBoost
     - Random Forest
     - Logistic Regression
     - LightGBM
   - Implemented a two-level prediction framework:
     - **Level 1:** Predict whether a patient will be admitted or not.
     - **Level 2:** Predict the specific non-admission category.

4. **Ensemble Learning**
   - Combined CatBoost and LightGBM to improve prediction performance.
   - Applied confidence-gating to improve robustness across different outcome classes.

5. **Temporal Prediction**
   - Developed a Deep Kalman Filter model to capture changes in patient conditions over time using multiple referrals.
   - Built a temporal prediction model capable of identifying referral patterns and improving long-term prediction accuracy.

The final system achieved an overall accuracy of **66.9%** for the ensemble model and **69.26%** for the temporal predictive model, providing AIC with a scalable decision-support tool for identifying patients who are more likely to accept community care services.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Deep Kalman Filter
- Matplotlib
