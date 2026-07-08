# Patient-Acceptance-Prediction

## Problem Statement
To develop a predictive model using referral data to analyze patterns, identify key factors and generate actionable insights for effective decision-making

## Objectives
- Which factors determine whether patients are more likely to accept or reject services
- Whether access to community-based services delay the need for residential care
- Which factors in patients that utilize community care delay the need for residential care

## Data Flow
1. `Leakage check.py`
   - validate the input data and optimization model to detect any data leakage or inconsistencies before analysis.
2. `Filtering.py`
   - remove invalid or irrelevant records and retain only feasible vessels, weather scenarios, and scheduling information required for optimization.
3. `Filtering test.py`
   - Execute the optimization model under multiple weather scenarios.
   - Evaluate vessel schedules, total operational cost, berth utilization, and Just-in-Time (JIT) arrival performance.

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
