# 🛵 Food Delivery Time Analytics & Prediction

A end-to-end data science project that analyses **45,584 Zomato food delivery orders** from India and builds a machine learning model to predict delivery time. Built as a student portfolio project demonstrating the full pipeline: data cleaning, feature engineering, model training, evaluation, and deployment as an interactive Streamlit web application.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Technologies Used](#5-technologies-used)
6. [Data Cleaning](#6-data-cleaning)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
8. [Feature Engineering](#8-feature-engineering)
9. [Machine Learning Approach](#9-machine-learning-approach)
10. [Evaluation Metrics and Results](#10-evaluation-metrics-and-results)
11. [Streamlit Application Features](#11-streamlit-application-features)
12. [AI Explanation Feature](#12-ai-explanation-feature)
13. [Project Architecture](#13-project-architecture)
14. [Installation and Setup](#14-installation-and-setup)
15. [API Key / Environment Setup](#15-api-key--environment-setup)
16. [Limitations](#16-limitations)
17. [Future Improvements](#17-future-improvements)

---

## 1. Project Overview

This project takes a raw food delivery dataset and walks through every stage of a data science workflow:

- **Data cleaning** — fixing missing values, corrupt timestamps, GPS coordinate errors, and a consistent city-name typo across 45,000+ rows
- **Feature engineering** — deriving 17 model-ready numeric features from raw text, GPS coordinates, and timestamps
- **Model training** — comparing five regression models and selecting the best based on held-out test performance
- **Evaluation** — measuring MAE, RMSE, and R² on a 20 % test split with an explicit overfitting check
- **Deployment** — a four-page Streamlit dashboard with live predictions and optional AI-generated explanations

---

## 2. Problem Statement

Food delivery platforms need accurate time estimates to manage customer expectations and rider scheduling. The raw dataset contains many factors that *may* influence delivery time — traffic density, weather, distance, rider characteristics, city type — but their relative importance is not obvious.

**Can we build a regression model that reliably predicts delivery time in minutes, given information known at the moment an order is placed?**

---

## 3. Objectives

1. Clean and validate a real-world delivery dataset with multiple quality issues.
2. Engineer informative numeric features from GPS coordinates, timestamps, and categorical columns.
3. Train and compare multiple regression models (baseline linear models and tree-based ensembles).
4. Evaluate models using MAE, RMSE, and R², and check explicitly for overfitting.
5. Identify the features most strongly associated with delivery time.
6. Deploy the best model as an interactive Streamlit application with live predictions.
7. Optionally enhance the prediction with a plain-language AI explanation via IBM watsonx.ai or OpenAI.

---

## 4. Dataset Description

| Property | Detail |
|---|---|
| **Source** | Zomato Food Delivery Dataset (Kaggle) |
| **Raw size** | 45,593 rows × 20 columns |
| **Cleaned size** | 45,584 rows × 20 columns (9 fully duplicate rows removed) |
| **Period** | February – April 2022 |
| **Geography** | India — three city types: Urban, Metropolitan, Semi-Urban |
| **Target variable** | `Time_taken (min)` — integer, range 10–54 minutes |

### Key columns

| Column | Type | Description |
|---|---|---|
| `Time_taken (min)` | Integer | **Target** — actual delivery duration in minutes |
| `Delivery_person_Age` | Integer | Rider's age (valid range: 18–50) |
| `Delivery_person_Ratings` | Float | Rider's average customer rating (1.0–5.0) |
| `Restaurant_latitude/longitude` | Float | GPS coordinates of the restaurant |
| `Delivery_location_latitude/longitude` | Float | GPS coordinates of the customer |
| `Road_traffic_density` | Categorical | Low / Medium / High / Jam |
| `Weather_conditions` | Categorical | Sunny / Cloudy / Windy / Fog / Sandstorms / Stormy |
| `City` | Categorical | Urban / Metropolitan / Semi-Urban |
| `Type_of_vehicle` | Categorical | motorcycle / scooter / electric_scooter / bicycle |
| `Type_of_order` | Categorical | Meal / Snack / Drinks / Buffet |
| `multiple_deliveries` | Integer | Number of simultaneous orders (0–3) |
| `Vehicle_condition` | Integer | Condition score 0 (poor) – 3 (excellent) |
| `Festival` | Binary | Whether a festival period is active (Yes / No) |
| `Order_Date` | Date | Date the order was placed |
| `Time_Orderd` | Time | Time the customer placed the order |
| `Time_Order_picked` | Time | Time the rider picked up from the restaurant |

---

## 5. Technologies Used

| Category | Library / Tool | Version |
|---|---|---|
| Language | Python | 3.x |
| Data manipulation | pandas | 2.2.2 |
| Numerical computing | numpy | 1.26.4 |
| Machine learning | scikit-learn | 1.4.2 |
| Model persistence | joblib | 1.4.2 |
| Visualisation | matplotlib | 3.9.0 |
| Visualisation | seaborn | 0.13.2 |
| Interactive charts | plotly | ≥ 5.20.0 |
| Web application | Streamlit | ≥ 1.35.0 |
| AI explanations (optional) | openai SDK | ≥ 1.30.0 |
| AI explanations (optional) | IBM watsonx.ai | REST API |

---

## 6. Data Cleaning

Implemented in [`src/data_cleaning.py`](src/data_cleaning.py). Run once before any other step.

The raw CSV contained several quality issues that required handling before the data could be used:

| Step | Issue | Fix applied |
|---|---|---|
| 1 | Literal string `"NaN"` used for missing values | Replaced with `np.nan` so pandas recognises them as null |
| 2 | Duplicate rows | Dropped fully duplicate rows (9 found) |
| 3 | `Delivery_person_Age` stored as a string object; ages < 18 present | Cast to numeric; set invalid ages to NaN |
| 4 | `Delivery_person_Ratings` had 53 values of 6.0 (above the 1–5 scale) | Set values > 5 to NaN |
| 5 | Restaurant GPS coordinates of `(0.0, 0.0)` (null-island) | Set those rows' restaurant and delivery coordinates to NaN |
| 6 | Negative latitude/longitude values (India coordinates must be positive) | Took absolute value |
| 7 | `Order_Date` stored as `"DD-MM-YYYY"` string | Parsed to `datetime` with `pd.to_datetime` |
| 8 | ~3,638 `Time_Orderd` values were Excel decimal serials (e.g. `0.875`) instead of `HH:MM` | Converted: `serial × 24 × 60 → HH:MM` |
| 9 | ~5,007 `Time_Order_picked` values were Excel serials; also `"24:05:00"` overflow entries | Converted serials; rolled `24:xx` to `00:xx` |
| 10 | City column had `"Metropolitian"` (misspelling) throughout | Standardised to `"Metropolitan"` |
| 11 | Remaining missing values in numeric and categorical columns | Numeric → median imputation; Categorical → mode imputation |

**Output:** `data/processed/zomato_cleaned.csv` — 45,584 rows × 20 columns, zero remaining nulls in model-relevant columns.

---

## 7. Exploratory Data Analysis

EDA scripts are in [`notebooks/eda.py`](notebooks/eda.py) and [`notebooks/feature_analysis.py`](notebooks/feature_analysis.py). Key findings (computed from live data — not hard-coded):

- **Traffic is the strongest categorical predictor.** Jam-traffic deliveries average significantly more minutes than low-traffic deliveries. Pearson r ≈ 0.41 with delivery time.
- **Distance is the strongest continuous predictor.** Haversine distance has a positive linear relationship with delivery time and ranks 3rd by permutation importance.
- **Multiple simultaneous deliveries increase time substantially.** Solo deliveries are faster than 3-order batches.
- **Higher-rated riders are associated with faster deliveries**, though this may reflect route assignment patterns rather than direct causation.
- **Weather encoding anomaly:** The ordinal weather scale (Sunny=1 to Stormy=6) does not perfectly match observed mean delivery times — Cloudy conditions show higher average times than Stormy, suggesting the severity ordering is imprecise.
- **Target distribution:** `Time_taken (min)` is integer-valued from 10 to 54 minutes. Sub-minute precision is not meaningful for this dataset.

---

## 8. Feature Engineering

Implemented in [`src/features.py`](src/features.py). 17 numeric features are derived from the cleaned dataset. The raw GPS, timestamp, and categorical string columns are replaced by these engineered features.

| # | Feature | Source | Rationale |
|---|---|---|---|
| 1 | `distance_km` | GPS lat/lon | Haversine great-circle distance (km) between restaurant and customer |
| 2 | `order_hour` | `Time_Orderd` | Hour of day (0–23) — captures lunch/dinner rush patterns |
| 3 | `order_day_of_week` | `Order_Date` | 0=Monday … 6=Sunday — captures weekly demand cycles |
| 4 | `order_month` | `Order_Date` | 2=Feb, 3=Mar, 4=Apr — captures monthly seasonality |
| 5 | `is_weekend` | `Order_Date` | Binary: 1 if Saturday or Sunday |
| 6 | `is_peak_hour` | `Time_Orderd` | Binary: 1 if lunch rush (11–14) or dinner rush (18–22) |
| 7 | `pickup_wait_min` | `Time_Orderd`, `Time_Order_picked` | Minutes between order placement and rider pickup from restaurant |
| 8 | `Delivery_person_Age` | Original column | Numeric age retained directly |
| 9 | `Delivery_person_Ratings` | Original column | Numeric rating retained directly |
| 10 | `multiple_deliveries` | Original column | Number of simultaneous deliveries (0–3) |
| 11 | `Vehicle_condition` | Original column | Condition score 0–3 retained directly |
| 12 | `traffic_encoded` | `Road_traffic_density` | Ordinal: Low=1, Medium=2, High=3, Jam=4 |
| 13 | `weather_encoded` | `Weather_conditions` | Ordinal severity: Sunny=1 … Stormy=6 |
| 14 | `city_encoded` | `City` | Ordinal: Urban=1, Metropolitan=2, Semi-Urban=3 |
| 15 | `vehicle_encoded` | `Type_of_vehicle` | Label: electric_scooter=0, motorcycle=1, scooter=2, bicycle=3 |
| 16 | `order_type_encoded` | `Type_of_order` | Label: Buffet=0, Drinks=1, Meal=2, Snack=3 |
| 17 | `festival_encoded` | `Festival` | Binary: No=0, Yes=1 |

**No data leakage:** `Time_taken (min)` is never used to construct any feature. All features use only information known at the moment an order is placed.

**Output:** `data/processed/zomato_features.csv` — 45,584 rows × 18 columns (17 features + target).

---

## 9. Machine Learning Approach

Implemented in [`src/train.py`](src/train.py).

### Problem type
Supervised regression — predict a continuous numeric target (`Time_taken (min)`).

### Train / test split
80 % training (36,467 rows) / 20 % test (9,117 rows), `random_state=42`.

### Models trained

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline. Wrapped in `Pipeline` with `StandardScaler` (linear models are sensitive to feature scale). |
| **Ridge Regression** | Linear baseline with L2 regularisation (`alpha=1.0`) to reduce overfitting on correlated features. Also scaled. |
| **Decision Tree** | Tree-based; captures non-linear patterns. `max_depth=10`, `min_samples_leaf=20` to limit overfitting. |
| **Random Forest** | Ensemble of 200 decision trees. `max_depth=15`, `min_samples_leaf=10`, `n_jobs=-1`. |
| **Gradient Boosting** | Sequential boosted trees. `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`. |

### Overfitting check
Train and test MAE are compared for each model. A gap > 1.5 minutes (test worse than train) is flagged as overfitting. All five models passed this check.

### Cross-validation
5-fold cross-validation was run on the two best-performing models (Random Forest and Gradient Boosting) using the full dataset. This provides a more robust performance estimate than a single split.

---

## 10. Evaluation Metrics and Results

### What the metrics mean

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** (Mean Absolute Error) | avg(|predicted − actual|) | Average error in minutes. Easy to explain: MAE = 3.3 min means predictions are off by about 3 minutes on average. Lower is better. |
| **RMSE** (Root Mean Squared Error) | √avg((predicted − actual)²) | Like MAE but large errors are penalised more heavily (squared). More sensitive to outliers than MAE. Lower is better. |
| **R²** (Coefficient of Determination) | 1 − SS_res / SS_tot | Proportion of variance in delivery time explained by the model. Range: (−∞, 1.0]. 1.0 = perfect, 0.0 = no better than predicting the mean. Higher is better. |

### Full model comparison (test set)

| Model | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² | Overfit |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **2.801** | **3.265** | **3.571** | **4.138** | **0.855** | **0.806** | **NO** |
| Gradient Boosting | 3.277 | 3.399 | 4.119 | 4.274 | 0.807 | 0.793 | NO |
| Decision Tree | 3.420 | 3.569 | 4.365 | 4.543 | 0.784 | 0.766 | NO |
| Ridge Regression | 5.130 | 5.157 | 6.476 | 6.507 | 0.524 | 0.520 | NO |
| Linear Regression | 5.130 | 5.157 | 6.476 | 6.507 | 0.524 | 0.520 | NO |

### Best model: Random Forest

```
Test MAE  = 3.265 min   → on average, predictions are off by ~3.3 minutes
Test RMSE = 4.138 min   → larger errors (e.g. rare outliers) penalised more
Test R²   = 0.806       → the model explains 80.6 % of variance in delivery time
Overfitting: NO         → train/test MAE gap = 0.46 min (well below 1.5 min threshold)
```

The best model is saved to `models/best_model.joblib` and its metadata (feature list, metric values) to `models/model_metadata.json`.

---

## 11. Streamlit Application Features

Run the dashboard with:

```bash
streamlit run app.py
```

The app has four pages, navigated from the sidebar:

### 🏠 Overview
- Hero banner with project description and live dataset statistics
- Four KPI cards: Total Deliveries, Average Delivery Time, Average Rider Rating, Average Distance
- Delivery time distribution histogram with mean line
- Orders-by-category pie charts (Traffic, City, Vehicle)
- Daily order volume trend line

### 📊 Delivery Analytics
Seven interactive Plotly charts organised into sections:
- **Operational Factors:** Delivery time by Traffic Density · by Weather Condition
- **Vehicle & Order Type:** Delivery time by Vehicle Type · by Order Type
- **City & Workload:** Delivery time by City Type · by Simultaneous Deliveries
- **Continuous Relationships:** Distance vs Delivery Time scatter (with linear trendline) · Rider Rating vs Delivery Time bubble chart

All charts show mean ± 95 % confidence intervals where applicable.

### 🤖 Prediction
- Input form with 13 user-controlled parameters across four sections: Route, Conditions, Rider, Order & Timing
- Live inference from the saved Random Forest model (`models/best_model.joblib`)
- Predicted time displayed in a styled result card with a ± MAE confidence range
- Feature importance bar chart showing the top 10 features by Random Forest MDI
- Model interpretation note with actual test metrics
- Optional AI plain-language explanation (see [Section 12](#12-ai-explanation-feature))

### 💡 Insights & Methodology
- Model performance comparison chart (Train vs Test MAE for all 5 models)
- R² comparison bar chart across all models
- Feature importance chart (Random Forest MDI, top 10)
- Three metric explanation cards: MAE, RMSE, R² with values from the actual trained model
- Key findings cards (6 EDA insights, computed dynamically from the loaded data)
- Limitations and caveats
- Project methodology — 6-step pipeline walkthrough
- About section

---

## 12. AI Explanation Feature

The AI explanation is **entirely optional**. The Random Forest model produces the numeric prediction independently. If no AI credentials are configured, the prediction still works — only the plain-language explanation section is hidden.

When configured, after a prediction is made the app calls an LLM with the order details and asks it to write 3–5 sentences explaining:
1. The main factors that contributed to the estimate
2. Any conditions that may increase or decrease the time
3. A reminder that this is a model-based estimate, not a guarantee

### Supported backends

| Backend | Credential required |
|---|---|
| **IBM watsonx.ai** | `api_key`, `project_id`, `base_url` |
| **OpenAI** (or compatible endpoint) | `api_key` |

Credentials are loaded from `.streamlit/secrets.toml` (Streamlit Cloud) or environment variables (local). **API keys are never stored in source code.**

---

## 13. Project Architecture

```
ibm_project/
│
├── data/
│   ├── raw/
│   │   └── Zomato Dataset.csv          ← Original unmodified source file
│   └── processed/
│       ├── zomato_cleaned.csv          ← Output of src/data_cleaning.py
│       └── zomato_features.csv         ← Output of src/features.py
│
├── models/
│   ├── best_model.joblib               ← Saved Random Forest (output of src/train.py)
│   ├── model_metadata.json             ← Feature names + test metrics (JSON)
│   └── model_comparison.csv            ← All 5 models compared (CSV)
│
├── notebooks/
│   ├── eda.py                          ← Exploratory data analysis plots
│   ├── model_evaluation.py             ← Evaluation plots (actual vs predicted, residuals)
│   └── feature_analysis.py             ← Feature importance, PDPs, correlation analysis
│
├── src/
│   ├── __init__.py                     ← Package stub
│   ├── data_cleaning.py                ← Step 1: cleaning pipeline
│   ├── features.py                     ← Step 2: feature engineering pipeline
│   ├── train.py                        ← Step 3: model training and evaluation
│   └── ai_explanation.py              ← Optional: LLM explanation module
│
├── .streamlit/
│   └── secrets.toml                    ← Credentials template (excluded from git)
│
├── app.py                              ← Streamlit application (entry point)
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Excludes secrets.toml, __pycache__, etc.
└── README.md                           ← This file
```

### Pipeline execution order

```
src/data_cleaning.py   →   src/features.py   →   src/train.py   →   streamlit run app.py
       ↓                         ↓                      ↓
zomato_cleaned.csv       zomato_features.csv     best_model.joblib
                                                 model_metadata.json
                                                 model_comparison.csv
```

---

## 14. Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

**1. Clone or download the repository**

```bash
git clone <your-repo-url>
cd ibm_project
```

**2. Create and activate a virtual environment** (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the data pipeline** (only needed once, or after changing source data)

```bash
python src/data_cleaning.py
python src/features.py
python src/train.py
```

Each script prints a detailed step-by-step log. The final line of `src/train.py` prints the best model name, test metrics, and saved file path.

**5. Launch the Streamlit app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 15. API Key / Environment Setup

The AI explanation feature is optional. If you skip this step, the app works fully — the prediction, all charts, and all analysis pages are unaffected.

### Option A — Streamlit secrets file (recommended for local use and Streamlit Cloud)

Create the file `.streamlit/secrets.toml`:

```toml
# IBM watsonx.ai backend
[watsonx]
backend     = "watsonx"
api_key     = "your-ibm-cloud-api-key"
project_id  = "your-watsonx-project-id"
base_url    = "https://us-south.ml.cloud.ibm.com"

# --- OR --- OpenAI backend
[openai]
backend     = "openai"
api_key     = "your-openai-api-key"
```

> ⚠️ `.streamlit/secrets.toml` is listed in `.gitignore` and must **never** be committed to version control.

### Option B — Environment variables

```bash
# IBM watsonx.ai
export WATSONX_API_KEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
export WATSONX_BASE_URL="https://us-south.ml.cloud.ibm.com"

# OR OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

The AI status (connected / not configured / error) is shown in the sidebar. If credentials are missing or invalid, the rest of the app continues to work normally.

---

## 16. Limitations

| Limitation | Detail |
|---|---|
| **Integer target resolution** | `Time_taken (min)` is recorded as whole-number minutes (range 10–54). Sub-minute precision is not meaningful for this dataset. |
| **Geographic scope** | Data covers Indian cities only (February–April 2022). The model may not generalise to other countries, platforms, or seasons without retraining. |
| **Weather encoding imprecision** | The ordinal weather encoding (Sunny=1 to Stormy=6) does not match the observed mean-time ranking in the data — Cloudy conditions show higher average delivery times than Stormy conditions. A target-encoded or one-hot representation may be more accurate. |
| **No live real-time inputs** | The model uses features known at order placement time. It has no access to real-time GPS tracking, live traffic APIs, or restaurant queue depth. |
| **Straight-line distance only** | `distance_km` is computed as a Haversine (great-circle) distance. Actual road distance would be a better predictor but requires a routing API. |
| **Correlation, not causation** | All relationships described in the app are predictive associations found in the data. They are not proven causal effects. |
| **Static model** | The model is trained once on historical data. It does not update as new deliveries are recorded. |

---

## 17. Future Improvements

| Area | Improvement |
|---|---|
| **Distance feature** | Replace Haversine with real road distance using Google Maps or OpenRouteService API |
| **Weather encoding** | Use target encoding or one-hot encoding for weather to better reflect its observed effect |
| **Hyperparameter tuning** | Apply `GridSearchCV` or `RandomizedSearchCV` to tune Random Forest and Gradient Boosting parameters |
| **Additional models** | Evaluate XGBoost, LightGBM, or a neural network; compare against current best |
| **Temporal validation** | Use a time-based train/test split (e.g. train on Feb–Mar, test on Apr) to simulate real deployment more accurately |
| **Real-time data** | Integrate a live traffic API to update the `traffic_encoded` feature dynamically at prediction time |
| **Prediction intervals** | Report a proper quantile regression interval instead of the ± MAE heuristic |
| **Model retraining pipeline** | Add a scheduled retraining script that incorporates new delivery data as it becomes available |
| **Authentication** | Add user authentication to the Streamlit app for deployment in a shared environment |

---

## Acknowledgements

- **Dataset:** Zomato Food Delivery Dataset — sourced from Kaggle
- **Libraries:** scikit-learn, pandas, numpy, Streamlit, Plotly, joblib
- **AI backends (optional):** IBM watsonx.ai (Granite 3.3 8B Instruct), OpenAI

---

*Student Data Science Portfolio Project · Python · scikit-learn · Streamlit*
