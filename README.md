# LogiSurge: End-to-End Delivery SLA Predictor 🚚

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://logisurge-sla-prediction-zplsvxgrwfdvzplj5rf6wh.streamlit.app])
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.3-orange)](https://xgboost.readthedocs.io/)

## Overview
LogiSurge is a full-pipeline machine learning application designed to predict the estimated delivery time (SLA) for logistics and quick-commerce operations. Instead of relying on static averages, this engine dynamically calculates delivery times using real-time features such as traffic density, weather conditions, delivery person metrics, and Haversine distance.

This repository demonstrates a rigorous, end-to-end data science workflow: from exploratory data analysis and leakage-safe data imputation to hyperparameter tuning, SHAP interpretability, and interactive deployment via Streamlit.

## Live Application
Access the live interactive deployment here: **[https://logisurge-sla-prediction-zplsvxgrwfdvzplj5rf6wh.streamlit.app]**

*Note: The app includes a "Load Random Sample" feature that pulls representative real-world data from the test set, automatically processes it through the live encoding pipeline, and generates a prediction.*

## Technical Pipeline & Data Science Rigor

### 1. Data Integrity & Cleaning
* **Fake-Null Handling:** Programmatically stripped string artifacts (e.g., `"conditions Sunny"` to `"Sunny"`) and mapped literal string `NaN`s to standard `np.nan`.
* **Anomaly Capping:** Enforced real-world logical bounds on the dataset (e.g., capping delivery person age to logical minimums, ratings to 5.0, and maximum batched deliveries to 3).

### 2. Leakage-Safe Imputation
* Missing timestamp data (Order Time and Pickup Time) was mathematically imputed using the **median preparation time**.
* *Crucially*, the median preparation time was calculated **strictly from the training fold** to prevent any data leakage from the validation/test sets.

### 3. Feature Engineering
The raw data was transformed to capture spatial and temporal patterns:
* **Spatial:** Computed the last-mile delivery distance using the **Haversine formula** based on exact latitude/longitude coordinates.
* **Temporal:** Implemented **Cyclic Time Encoding** (Sine and Cosine transformations) for `Time_Orderd` to help the model understand the cyclical nature of a 24-hour clock. 
* **Categorical:** Derived `Day_of_Week` and `Is_Weekend` flags, applied ordinal encoding for hierarchical variables (traffic, weather), and one-hot encoding for nominal variables.

### 4. Model Training & Tuning
* **Algorithm:** XGBoost Regressor.
* **Validation Strategy:** 80/20 Train-Validation split (Trained on 36,474 rows, validated on 9,119 rows).
* **Hyperparameter Tuning:** Conducted a `RandomizedSearchCV` with 3-fold cross-validation over 10 iterations to optimize for MAE.
* **Optimal Parameters:** `learning_rate=0.01`, `max_depth=11`, `n_estimators=600`, `subsample=0.7`, `colsample_bytree=0.8`.

## Honest Model Performance
The tuned model achieved the following metrics on the held-out validation set:
* **Mean Absolute Error (MAE):** 3.16 minutes
* **R-squared Score (R²):** 0.8215

These metrics indicate that the model explains ~82% of the variance in delivery times and its predictions are, on average, within ~3.16 minutes of the actual delivery time.

## Interpretability (SHAP)
To ensure the model is not a black box, SHAP (SHapley Additive exPlanations) TreeExplainer was utilized.
*(Include a screenshot of your SHAP summary plot or waterfall plot here to showcase feature importance. Example: `![SHAP Plot](images/shap_summary.png)`)*

## Local Setup & Installation

To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/DHRUV15019/LogiSurge-SLA-Prediction.git](https://github.com/DHRUV15019/LogiSurge-SLA-Prediction.git)
   cd LogiSurge-SLA-Prediction