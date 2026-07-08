# Patient-Acceptance-Prediction

## Problem Statement
To develop a predictive model using referral data to analyze patterns, identify key factors and generate actionable insights for effective decision-making

## Objectives
- Which factors determine whether patients are more likely to accept or reject services
- Whether access to community-based services delay the need for residential care
- Which factors in patients that utilize community care delay the need for residential care

## Data Flow
The usual flow is:
1. `Leakage check.py`
   - validate the input data and optimization model to detect any data leakage or inconsistencies before analysis.
2. `Filtering.py`
   - remove invalid or irrelevant records and retain only feasible vessels, weather scenarios, and scheduling information required for optimization.
3. `Filtering test.py`
   - Execute the optimization model under multiple weather scenarios.
   - Evaluate vessel schedules, total operational cost, berth utilization, and Just-in-Time (JIT) arrival performance.
