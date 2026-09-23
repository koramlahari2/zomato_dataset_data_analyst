"""Generate the academic project report from the current project artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "notebooks" / "plots"
OUTPUT = ROOT / "Food_Delivery_Analytics_AI_Project_Report.pdf"
FONT = "DejaVu Serif"
SANS = "DejaVu Sans"


def load_artifacts() -> dict:
    clean = pd.read_csv(ROOT / "data" / "processed" / "zomato_cleaned.csv")
    features = pd.read_csv(ROOT / "data" / "processed" / "zomato_features.csv")
    raw = pd.read_csv(ROOT / "data" / "raw" / "Zomato Dataset.csv")
    comparison = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    importance = pd.DataFrame({
        "feature": metadata["feature_names"],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return {
        "raw": raw, "clean": clean, "features": features,
        "comparison": comparison, "metadata": metadata,
        "model": model, "importance": importance,
    }


def page(pdf: PdfPages, title: str | None = None, subtitle: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    if title:
        fig.text(0.09, 0.935, title, fontsize=19, fontweight="bold",
                 color="#18324b", family=FONT)
        fig.add_artist(plt.Line2D([0.09, 0.91], [0.915, 0.915],
                                  color="#c47b38", linewidth=1.4))
    if subtitle:
        fig.text(0.09, 0.885, subtitle, fontsize=9.5, color="#5c6872",
                 family=SANS)
    return fig


def footer(fig, number: int):
    fig.text(0.09, 0.035, "Food Delivery Analytics and AI-Based Delivery Time Prediction",
             fontsize=7.5, color="#687581", family=SANS)
    fig.text(0.91, 0.035, str(number), ha="right", fontsize=8,
             color="#687581", family=SANS)


def text_block(fig, text: str, x=0.10, y=0.84, width=0.80, size=10.5,
               color="#253746", line_height=1.45):
    fig.text(x, y, text, va="top", fontsize=size, color=color, family=FONT,
             linespacing=line_height, wrap=True, transform=fig.transFigure)


def bullets(fig, items: list[str], x=0.11, y=0.82, size=10.5, gap=0.047):
    for i, item in enumerate(items):
        fig.text(x, y - i * gap, "\u2022 " + item, va="top", fontsize=size,
                 color="#253746", family=FONT, wrap=True)


def table(fig, data, columns, x=0.10, y=0.82, width=0.80, row_height=0.034,
          fontsize=8.3, header=True):
    ax = fig.add_axes([x, y - row_height * (len(data) + 1), width,
                       row_height * (len(data) + 1)])
    ax.axis("off")
    cell = ax.table(cellText=data, colLabels=columns if header else None,
                    cellLoc="left", colLoc="left", loc="upper left",
                    bbox=[0, 0, 1, 1])
    cell.auto_set_font_size(False)
    cell.set_fontsize(fontsize)
    for (r, c), obj in cell.get_celld().items():
        obj.set_edgecolor("#d5dce1")
        obj.PAD = 0.015
        if header and r == 0:
            obj.set_facecolor("#18324b")
            obj.get_text().set_color("white")
            obj.get_text().set_weight("bold")
        else:
            obj.set_facecolor("#f4f7f8" if r % 2 else "white")
    return ax


def image_page(pdf, title, image_names, captions, number):
    fig = page(pdf, title, "Figures reproduced from the project's generated analysis artifacts.")
    n = len(image_names)
    heights = [0.33] * n if n <= 3 else [0.25] * n
    top = 0.84
    for i, (name, caption) in enumerate(zip(image_names, captions)):
        h = heights[i]
        ax = fig.add_axes([0.11, top - h, 0.78, h - 0.035])
        ax.imshow(mpimg.imread(PLOTS / name))
        ax.axis("off")
        fig.text(0.11, top - h - 0.018, caption, fontsize=8.5,
                 color="#364a58", family=FONT, style="italic")
        top -= h + 0.065
    footer(fig, number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def architecture_page(pdf, number):
    fig = page(pdf, "10. System Architecture", "Verified project pipeline from source data to user-facing explanation.")
    labels = ["Dataset", "Data Cleaning", "Exploratory\nData Analysis", "Feature\nEngineering",
              "Machine Learning\nModel", "Delivery Time\nPrediction", "Streamlit\nDashboard", "AI\nExplanation"]
    ys = np.linspace(0.80, 0.18, len(labels))
    for i, (label, y) in enumerate(zip(labels, ys)):
        box = FancyBboxPatch((0.26, y - 0.032), 0.48, 0.06,
                             boxstyle="round,pad=0.008", linewidth=1.2,
                             edgecolor="#18324b", facecolor="#eaf1f4")
        fig.add_artist(box)
        fig.text(0.50, y, label, ha="center", va="center", fontsize=10,
                 family=SANS, color="#18324b", fontweight="bold")
        if i < len(labels) - 1:
            fig.add_artist(FancyArrowPatch((0.50, y - 0.035),
                                           (0.50, ys[i + 1] + 0.035),
                                           arrowstyle="-|>", mutation_scale=12,
                                           linewidth=1.0, color="#c47b38"))
    footer(fig, number)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    a = load_artifacts()
    clean, raw, features = a["clean"], a["raw"], a["features"]
    meta, comparison, importance = a["metadata"], a["comparison"], a["importance"]
    pages = []
    with PdfPages(OUTPUT) as pdf:
        n = 1
        # Cover
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#18324b")
        fig.text(0.10, 0.78, "FOOD DELIVERY ANALYTICS", fontsize=13,
                 color="#e8b36e", family=SANS, fontweight="bold")
        fig.text(0.10, 0.68, "and AI-Based Delivery\nTime Prediction", fontsize=29,
                 color="white", family=FONT, fontweight="bold", linespacing=1.2)
        fig.text(0.10, 0.48, "Academic Project Report", fontsize=16,
                 color="#dce7ed", family=FONT)
        fig.text(0.10, 0.30, "Student Name: [MY NAME]\nDepartment: Computer Science and Engineering\n"
                 "Internship/Program: IBM SkillsBuild Data Analytics with AI\nInstitution: [COLLEGE NAME]\n"
                 "Academic Year: 2026\nProject Submission Date: [SUBMISSION DATE]",
                 fontsize=11.5, color="white", family=FONT, linespacing=1.7)
        fig.text(0.10, 0.08, "Prepared from the completed project files and generated results",
                 fontsize=8.5, color="#b8cbd6", family=SANS)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Certificate
        fig = page(pdf, "CERTIFICATE")
        text_block(fig, "This is to certify that [STUDENT NAME], of the Department of Computer Science and Engineering at [COLLEGE NAME], has completed the academic project titled \"Food Delivery Analytics and AI-Based Delivery Time Prediction\" as part of the IBM SkillsBuild Data Analytics with AI internship/project work.", y=0.78, size=12)
        text_block(fig, "The work presented in this report is based on the project files, processed dataset, trained model, analysis outputs, and Streamlit application available in the submitted workspace.", y=0.56, size=11)
        fig.text(0.12, 0.27, "Guide/Mentor: ______________________________\n\nSignature: __________________________________\n\nDate: ______________________________________\n\nInstitution Seal: ___________________________", fontsize=11, family=FONT, color="#253746", linespacing=1.6)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Formal pages
        for title, body in [
            ("DECLARATION", "I, [STUDENT NAME], declare that the project report titled \"Food Delivery Analytics and AI-Based Delivery Time Prediction\" is prepared from the work completed during the IBM SkillsBuild Data Analytics with AI internship/project program. The analysis, model results, application description, and figures in this report are derived from the project artifacts available in the workspace. Any external or project-provided source is identified in the References section.\n\nPlace: [PLACE]\nDate: [DATE]\n\nSignature: ______________________________"),
            ("ACKNOWLEDGEMENT", "I express my sincere gratitude to the IBM SkillsBuild Data Analytics with AI program for providing the learning context and project opportunity for this work. I thank my guide/mentor, faculty members, and institution for their guidance and support. I also acknowledge the open-source Python ecosystem and the dataset attribution recorded in the project README.\n\nThis report was prepared from the completed data-cleaning pipeline, feature-engineering pipeline, model-training outputs, exploratory-analysis figures, and Streamlit application."),
            ("ABSTRACT", "This project develops an end-to-end workflow for analysing food-delivery orders and estimating delivery time in minutes. The project uses the Zomato Food Delivery Dataset available in the workspace, containing 45,584 records and 20 columns in the current raw and cleaned files. Data preparation addresses missing representations, duplicate checks, invalid rider values, coordinate issues, date/time formats, categorical normalization, and imputation. Seventeen numeric features are produced for supervised regression. Five regression models are compared on a fixed 80/20 split, and the saved Random Forest model achieves a test MAE of 3.265 minutes, test RMSE of 4.138 minutes, and test R² of 0.8059. A Streamlit dashboard provides overview analytics, interactive charts, prediction, and methodology views. An optional AI integration uses an LLM only to explain a model prediction in plain language; the Random Forest remains responsible for the numeric estimate."),
        ]:
            fig = page(pdf, title); text_block(fig, body, y=0.82, size=11.5); footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # TOC
        fig = page(pdf, "TABLE OF CONTENTS")
        toc = ["1. Introduction", "2. Dataset Description", "3. Technologies Used", "4. System Architecture", "5. Data Preprocessing", "6. Exploratory Data Analysis", "7. Feature Engineering", "8. Machine Learning", "9. Model Evaluation", "10. Model Interpretation", "11. Streamlit Application", "12. AI Integration", "13. Results and Findings", "14. Limitations", "15. Future Enhancements", "16. Conclusion", "17. References", "18. Appendix"]
        for i, item in enumerate(toc): fig.text(0.12, 0.82 - i * 0.039, item, fontsize=11, family=FONT, color="#253746")
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Intro
        fig = page(pdf, "1. INTRODUCTION")
        text_block(fig, "Background\nFood-delivery operations combine route distance, traffic, weather, rider conditions, order context, and workload. The project studies these recorded attributes and builds a reproducible regression workflow for delivery-time estimation.\n\nProblem Statement\nCan a supervised regression model estimate the delivery duration from information available around order placement, while an analytics dashboard communicates descriptive patterns and uncertainty?\n\nMotivation\nA data-driven estimate can support clearer customer expectations and operational analysis. The project also provides a complete student-scale example spanning cleaning, analysis, feature engineering, model comparison, interpretation, and deployment.\n\nObjectives\n1. Validate and clean the delivery records.\n2. Create numeric predictors from time, location, categorical, and rider fields.\n3. Compare five regression models using MAE, RMSE, and R².\n4. Deploy the saved best model in Streamlit.\n5. Add an optional natural-language explanation layer without confusing it with the predictive model.\n\nScope\nThe analysis is limited to the supplied Indian delivery dataset and its February-April 2022 period. It is a project demonstration, not a real-time dispatch system.", y=0.84, size=10.5)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Dataset
        fig = page(pdf, "2. DATASET DESCRIPTION")
        text_block(fig, f"Dataset source/name: Zomato Food Delivery Dataset (source attribution recorded in README). The current files contain {len(raw):,} raw records, {len(clean.columns)} cleaned columns, and {len(features.columns)} feature-dataset columns including the target. The target is `Time_taken (min)`, an integer from {clean['Time_taken (min)'].min()} to {clean['Time_taken (min)'].max()} minutes; mean={clean['Time_taken (min)'].mean():.3f}, median={clean['Time_taken (min)'].median():.1f}. Dates in the cleaned file span {clean['Order_Date'].min()} to {clean['Order_Date'].max()}.", y=0.84, size=10.5)
        rows = [["Raw CSV", f"{len(raw):,} x {len(raw.columns)}", "Current data/raw file"], ["Cleaned CSV", f"{len(clean):,} x {len(clean.columns)}", "Analytics dataframe"], ["Feature CSV", f"{len(features):,} x {len(features.columns)}", "17 features + target"], ["Target", "Time_taken (min)", "Integer minutes; 10-54"], ["Raw time missing values", "2,601 total", "Time_Orderd=2,161; Time_Order_picked=440"], ["Feature-file missing values", "0", "Model-ready output"]]
        table(fig, rows, ["Artifact / field", "Verified value", "Meaning"], y=0.62, row_height=0.045, fontsize=8.6)
        text_block(fig, "Important features include rider age and rating, GPS coordinates, traffic, weather, vehicle condition and type, order type, simultaneous deliveries, festival status, order date/time, and the target. The cleaned dataframe preserves some missing raw time entries, while the engineered feature dataset used for training contains no missing values.", y=0.28, size=10.2)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Technologies
        fig = page(pdf, "3. TECHNOLOGIES USED")
        tech = [["Python", "Project language and scripts"], ["pandas", "CSV loading, cleaning, grouping, feature tables"], ["NumPy", "Numeric operations and missing-value handling"], ["scikit-learn", "Splitting, preprocessing, regressors, metrics, validation"], ["matplotlib / seaborn", "EDA and model-evaluation plots"], ["Plotly", "Interactive Streamlit charts"], ["Streamlit", "Four-page dashboard and prediction interface"], ["joblib", "Persistence and loading of the trained model"], ["OpenAI SDK / IBM watsonx.ai REST", "Optional AI explanation backends; credentials externalized"]]
        table(fig, tech, ["Technology", "Verified use in project"], y=0.84, row_height=0.052, fontsize=9)
        text_block(fig, "The dependency file also lists Jupyter and Flask, but the completed application path documented here is the Python scripts, generated plots, saved model artifacts, and Streamlit app. The AI backends are optional and are not required for the numeric prediction.", y=0.30, size=10.5)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        architecture_page(pdf, n); n += 1

        # Preprocessing
        fig = page(pdf, "5. DATA PREPROCESSING")
        text_block(fig, "The cleaning pipeline in `src/data_cleaning.py` performs the following operations:\n\n1. Loads the raw CSV and replaces literal `NaN` strings with real nulls.\n2. Checks and removes fully duplicated records in the cleaning pipeline.\n3. Converts rider age and rating to numeric; invalid ages below 18 and ratings above 5 are set missing.\n4. Converts coordinate fields to numeric, removes null-island coordinates, and fixes negative Indian latitude/longitude signs.\n5. Parses `Order_Date` from day-month-year format.\n6. Converts Excel decimal time serials to `HH:MM`, and handles `24:xx` pickup overflow.\n7. Standardizes `Metropolitian` to `Metropolitan`.\n8. Imputes remaining numeric values with medians and categorical values with modes.\n9. Casts `multiple_deliveries` and rider age to integer-like types.\n\nThe current cleaned CSV retains missing values in the raw time columns, but the feature-building stage converts those fields into model features and the current feature CSV has zero missing values.", y=0.84, size=10.3)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # EDA narrative
        fig = page(pdf, "6. EXPLORATORY DATA ANALYSIS")
        text_block(fig, "The EDA scripts create descriptive plots from the cleaned data. These are associations in the observed dataset, not causal claims. The target is concentrated around the mid-20-minute range, with a verified mean of 26.294 minutes and median of 26.0 minutes. Traffic groups differ clearly: Low averages 21.464 minutes, Medium 26.700, High 27.240, and Jam 31.176. Weather averages range from 21.857 minutes for Sunny to 28.917 for Cloudy, so the assumed ordinal weather scale should be interpreted carefully.\n\nVehicle and order-type differences are smaller than the traffic and workload differences. Motorcycles average 27.606 minutes, scooters 24.479, electric scooters 24.470, and bicycles 26.426 across only 68 records. Order types range narrowly from 26.188 (Drinks) to 26.418 (Meal). City and simultaneous-delivery groups require sample-size caution: Semi-Urban has 164 records and averages 49.732 minutes, while three simultaneous deliveries has 361 records and averages 47.820 minutes.", y=0.84, size=10.6)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        image_page(pdf, "Figure Set A. EDA: Target, Weather, and Traffic", ["01_delivery_time_distribution.png", "02_delivery_time_by_weather.png", "03_delivery_time_by_traffic.png"], ["Figure 1. Overall delivery-time distribution with histogram/KDE and box plot.", "Figure 2. Delivery time by weather condition.", "Figure 3. Delivery time by road traffic density."], n); n += 1
        image_page(pdf, "Figure Set B. EDA: Vehicle, Order, City, and Workload", ["04_delivery_time_by_vehicle_type.png", "06_delivery_time_by_order_type.png", "07_delivery_time_by_city.png", "09_delivery_time_by_multiple_deliveries.png"], ["Figure 4. Delivery time by vehicle type; bicycle has a small group.", "Figure 5. Delivery time by order type; group means are close.", "Figure 6. Delivery time by city tier; Semi-Urban has a small sample.", "Figure 7. Delivery time by simultaneous deliveries."], n); n += 1
        image_page(pdf, "Figure Set C. EDA: Condition, Rating, Distance, and Interaction", ["05_delivery_time_by_vehicle_condition.png", "08_delivery_time_by_rating.png", "10_delivery_time_vs_distance.png", "11_heatmap_traffic_weather.png"], ["Figure 8. Delivery time by vehicle condition score.", "Figure 9. Delivery time by rider rating.", "Figure 10. Delivery time versus Haversine distance.", "Figure 11. Traffic-weather interaction heatmap."], n); n += 1

        # Features
        fig = page(pdf, "7. FEATURE ENGINEERING")
        rows = [["distance_km", "GPS coordinates", "Haversine straight-line distance"], ["order_hour", "Time_Orderd", "Hour 0-23"], ["order_day_of_week", "Order_Date", "Monday=0 through Sunday=6"], ["order_month", "Order_Date", "Calendar month"], ["is_weekend", "Order_Date", "Weekend indicator"], ["is_peak_hour", "Time_Orderd", "Lunch/dinner peak indicator"], ["pickup_wait_min", "Order and pickup times", "Minutes between order and pickup"], ["Age, rating, workload, condition", "Original numeric fields", "Retained numeric inputs"], ["traffic_encoded / weather_encoded / city_encoded", "Categorical fields", "Ordinal maps"], ["vehicle_encoded / order_type_encoded", "Categorical fields", "Label encodings"], ["festival_encoded", "Festival", "Binary encoding"]]
        table(fig, rows, ["Feature(s)", "Source", "Construction"], y=0.84, row_height=0.047, fontsize=8.25)
        text_block(fig, "The Haversine calculation uses latitude and longitude to estimate great-circle distance over the Earth. It is a straight-line distance, not a road-network distance. The final feature file contains 18 columns: 17 numeric predictors and the target. The target is not used to construct any feature, avoiding direct leakage from delivery duration.", y=0.22, size=10.3)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # ML
        fig = page(pdf, "8. MACHINE LEARNING")
        text_block(fig, "Problem type: supervised regression. Target: `Time_taken (min)`.\n\nThe feature file is split into 80% training data (36,467 rows) and 20% test data (9,117 rows) using `random_state=42`. Linear Regression and Ridge use StandardScaler pipelines. Decision Tree, Random Forest, and Gradient Boosting use tree-based models without feature scaling.\n\nModels tested:\n• Linear Regression baseline\n• Ridge Regression with alpha=1.0\n• Decision Tree Regressor with max_depth=10 and min_samples_leaf=20\n• Random Forest Regressor with 200 trees, max_depth=15, min_samples_leaf=10, random_state=42\n• Gradient Boosting Regressor with 200 estimators, learning_rate=0.05, max_depth=5, subsample=0.8\n\nThe saved best model is selected by lowest test MAE. Five-fold cross-validation is also run for Random Forest and Gradient Boosting in the training script.", y=0.84, size=10.8)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Evaluation
        fig = page(pdf, "9. MODEL EVALUATION")
        rows = comparison[["model", "train_mae", "test_mae", "train_rmse", "test_rmse", "train_r2", "test_r2", "overfit"]].round(4).astype(str).values.tolist()
        table(fig, rows, ["Model", "Train MAE", "Test MAE", "Train RMSE", "Test RMSE", "Train R²", "Test R²", "Overfit"], y=0.84, row_height=0.049, fontsize=7.8)
        text_block(fig, "MAE is the average absolute prediction error in minutes. RMSE weights larger errors more heavily. R² is the proportion of target variance explained by the model. Random Forest is the saved best model: test MAE=3.265 minutes, test RMSE=4.138 minutes, and test R²=0.8059. Its train/test MAE gap is 0.464 minutes, below the project's 1.5-minute overfitting flag threshold.", y=0.45, size=10.5)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1
        image_page(pdf, "Figure Set D. Model Evaluation", ["model_comparison_bar.png", "actual_vs_predicted.png", "residuals_distribution.png"], ["Figure 12. Test-set comparison of the five trained regressors.", "Figure 13. Actual versus predicted delivery time on the held-out test set.", "Figure 14. Residual distribution and residuals versus predicted values."], n); n += 1

        # Interpretation
        fig = page(pdf, "10. MODEL INTERPRETATION")
        top = importance.head(10).copy(); top["importance"] = (top["importance"] * 100).round(2).astype(str) + "%"
        table(fig, top.values.tolist(), ["Feature", "Random Forest MDI"], y=0.84, row_height=0.047, fontsize=9)
        text_block(fig, "The saved Random Forest's built-in mean-decrease-impurity (MDI) ranking places Delivery_person_Ratings first, followed by weather_encoded, traffic_encoded, multiple_deliveries, distance_km, rider age, and vehicle condition. These values are relative split-based importance scores that sum to approximately 100%; they are not causal effects and do not mean that changing a feature will change delivery time by the same proportion. The feature-analysis script also produces permutation-importance, partial-dependence, and correlation-versus-MDI views to compare model utility and descriptive association.", y=0.31, size=10.3)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1
        image_page(pdf, "Figure Set E. Feature Interpretation", ["feature_importance.png", "fa_mdi_vs_permutation.png", "fa_partial_dependence.png"], ["Figure 15. Saved-model feature importance chart.", "Figure 16. MDI versus held-out permutation importance.", "Figure 17. Partial-dependence profiles for top permutation-important features."], n); n += 1

        # Streamlit
        fig = page(pdf, "11. STREAMLIT APPLICATION")
        rows = [["Overview", "Live KPIs, distribution histogram, category pies, daily order trend"], ["Delivery Analytics", "Interactive Plotly charts for traffic, weather, vehicle, order, city, workload, distance, and rating"], ["Prediction", "13-input form, saved Random Forest prediction, MAE range, feature-importance chart, optional explanation"], ["Insights & Methodology", "Model comparison, R² chart, feature importance, metric cards, EDA findings, limitations, and pipeline overview"]]
        table(fig, rows, ["Page", "Verified features"], y=0.84, row_height=0.07, fontsize=9)
        text_block(fig, "The app loads the cleaned and feature datasets, saved model, metadata, and comparison table relative to `app.py`. It protects optional AI failures so that prediction and analytics remain available when no credentials are configured. The dashboard uses Plotly for interactive charts and displays the saved test metrics in the prediction and insights views.", y=0.46, size=10.5)
        text_block(fig, "Application evidence note: the live Streamlit application was opened during report preparation and its four navigation pages were verified. The integrated browser stores captures outside the project workspace, so this report does not embed unexported or fabricated screenshot files.", y=0.27, size=9.4, color="#5c6872")
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # AI
        fig = page(pdf, "12. AI INTEGRATION")
        text_block(fig, "The project separates prediction from explanation. The Random Forest model produces the numeric delivery-time estimate. The optional AI layer receives the prediction and order context, then asks an LLM to produce a short plain-language explanation. The prompt explicitly asks the model to describe associations rather than causal claims and to remind the user that the result is an estimate.\n\nSupported backends in `src/ai_explanation.py` are IBM watsonx.ai through REST and OpenAI or an OpenAI-compatible endpoint through the SDK. Credentials are loaded from Streamlit secrets or environment variables; they are not hard-coded in source and are not included in this report. The code returns safe status/error messages for missing configuration, authentication, rate limits, network failures, and provider errors.\n\nThe AI text does not alter the numeric prediction. When no credentials are configured, the application continues to provide the model prediction and analytics pages.", y=0.84, size=10.6)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Results
        fig = page(pdf, "13. RESULTS AND FINDINGS")
        bullets(fig, ["The current project artifacts contain 45,584 delivery records and a model-ready file with 17 predictors plus the target.", "Random Forest is the best saved model by test MAE: 3.265 minutes; test RMSE is 4.138 minutes and test R² is 0.8059.", "Observed traffic means increase from Low (21.464 minutes) to Jam (31.176 minutes), supporting traffic as an important descriptive grouping.", "Simultaneous-delivery means rise from 22.876 minutes for zero to 47.820 minutes for three, with the highest workload groups much smaller than the one-order group.", "Vehicle-type and order-type means are comparatively close; bicycle results use only 68 records and should be treated cautiously.", "The saved model's MDI ranking is led by rider rating, weather encoding, traffic encoding, simultaneous deliveries, distance, rider age, and vehicle condition.", "All findings are predictive associations in this dataset. They do not establish causal effects."])
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Limits/future
        fig = page(pdf, "14. LIMITATIONS AND 15. FUTURE ENHANCEMENTS")
        text_block(fig, "Limitations\n• The data represents Indian deliveries from February-April 2022 and may not generalize across geography, platform, or season.\n• The target is integer-valued from 10-54 minutes; sub-minute precision is not supported.\n• Haversine distance is straight-line distance rather than road distance.\n• Weather is encoded ordinally, although observed category means do not perfectly follow that assumed severity order.\n• The model has no live traffic, GPS, restaurant queue, or current weather inputs.\n• A single historical split and static saved model cannot guarantee future performance.\n• AI output is generated language and may be incomplete; it is not the source of the numeric prediction.\n\nFuture Enhancements\n• Add current traffic/weather and road-network distance.\n• Use temporal validation and scheduled monitoring/retraining.\n• Tune current models and evaluate additional boosted or neural models.\n• Produce formal prediction intervals rather than the dashboard's ±MAE display.\n• Add authentication and deployment monitoring for shared use.\n• Extend the dashboard with business metrics and route-level analysis.", y=0.84, size=10.4)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Conclusion/references
        fig = page(pdf, "16. CONCLUSION")
        text_block(fig, "The project demonstrates a complete, reproducible food-delivery analytics workflow. It begins with data validation and cleaning, creates model-ready numeric features, compares five regressors, saves a Random Forest model, and presents both descriptive analytics and prediction through Streamlit. The verified test metrics show a useful held-out estimate for this project dataset, while the analysis and limitations make clear that the model is not a real-time guarantee. The optional AI integration adds accessible language around a prediction without replacing the machine-learning model.", y=0.84, size=11)
        fig.text(0.10, 0.49, "17. REFERENCES", fontsize=17, fontweight="bold", color="#18324b", family=FONT)
        refs = ["[1] Project README.md, Food Delivery Time Analytics & Prediction, current workspace.", "[2] Project source scripts: src/data_cleaning.py, src/features.py, src/train.py, src/ai_explanation.py.", "[3] Project analysis scripts and generated figures: notebooks/eda.py, notebooks/feature_analysis.py, notebooks/model_evaluation.py.", "[4] Zomato Food Delivery Dataset, source attribution recorded in the project README and supplied data files.", "[5] Project application: app.py and requirements.txt, current workspace."]
        bullets(fig, refs, y=0.43, size=9.7, gap=0.055)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

        # Appendix tables
        fig = page(pdf, "18. APPENDIX A: VERIFIED TABLES")
        stats = []
        for col in ["Road_traffic_density", "Weather_conditions", "Type_of_vehicle", "Type_of_order", "City", "multiple_deliveries"]:
            for value, row in clean.groupby(col)["Time_taken (min)"].agg(["mean", "median", "count"]).round(3).iterrows():
                stats.append([col, str(value), f"{row['mean']:.3f}", f"{row['median']:.3f}", f"{int(row['count']):,}"])
        table(fig, stats[:18], ["Column", "Category", "Mean", "Median", "Count"], y=0.84, row_height=0.036, fontsize=7.2)
        text_block(fig, "Appendix table excerpt: grouped descriptive statistics calculated directly from data/processed/zomato_cleaned.csv. The remaining category rows are represented in the generated EDA figures and can be reproduced with the included notebooks/eda.py script.", y=0.12, size=8.8, color="#5c6872")
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1
        image_page(pdf, "Appendix B. Generated Analysis Figures", ["fa_correlation_scatter.png", "fa_group_means.png", "model_comparison_bar.png"], ["Figure 18. Pearson association versus Random Forest MDI importance.", "Figure 19. Group means for selected encoded/categorical features.", "Figure 20. Model comparison figure used by the evaluation workflow."], n); n += 1
        image_page(pdf, "Appendix C. Evaluation Figures", ["feature_importance.png", "actual_vs_predicted.png", "residuals_distribution.png"], ["Figure 21. Relative feature importance.", "Figure 22. Held-out actual versus predicted values.", "Figure 23. Residual diagnostic figure."], n); n += 1

        # App evidence / code provenance
        fig = page(pdf, "Appendix D. Application Evidence and Reproducibility")
        text_block(fig, "Verified Streamlit navigation\n• Overview: 45,584 delivery records, average delivery time, rider rating, Haversine distance, distribution, category pies, and daily volume.\n• Delivery Analytics: traffic, weather, vehicle, order, city, simultaneous deliveries, distance, and rider-rating views.\n• Prediction: route, conditions, rider, order/timing inputs; saved Random Forest inference; ± test-MAE display; model-level feature importance; optional AI explanation.\n• Insights & Methodology: model comparison, metric explanations, feature importance, EDA findings, limitations, and methodology.\n\nReproduction commands recorded in README\npython src/data_cleaning.py\npython src/features.py\npython src/train.py\nstreamlit run app.py\n\nSelected project outputs\ndata/processed/zomato_cleaned.csv\ndata/processed/zomato_features.csv\nmodels/best_model.joblib\nmodels/model_metadata.json\nmodels/model_comparison.csv\nnotebooks/plots/*.png\n\nNo API keys, secrets, passwords, or credential values are included in this report.", y=0.84, size=10.2)
        footer(fig, n); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig); n += 1

    print(f"Generated {OUTPUT} with {n - 1} pages")


if __name__ == "__main__":
    main()