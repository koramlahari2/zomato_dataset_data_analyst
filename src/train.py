"""
src/train.py
============
Machine learning pipeline for predicting food delivery time.

Run from the project root:
    python src/train.py

Input  : data/processed/zomato_features.csv
Output : models/best_model.joblib
         models/model_comparison.csv

Models trained
--------------
1. Linear Regression          — simple baseline, interpretable, assumes linear
                                 relationships between features and target.
2. Ridge Regression           — linear baseline with L2 regularisation to
                                 reduce overfitting on correlated features.
3. Decision Tree Regressor    — tree-based, captures non-linear patterns,
                                 prone to overfitting without tuning.
4. Random Forest Regressor    — ensemble of 200 decision trees, reduces
                                 variance vs a single tree, strong performer.
5. Gradient Boosting Regressor— boosted trees, fits residuals sequentially,
                                 often the best-performing single model.

Evaluation metrics used
-----------------------
  MAE   — Mean Absolute Error: average absolute difference between predicted
           and actual delivery time (in minutes). Easy to interpret.
           Lower is better. e.g. MAE=4.2 means predictions are off by ~4 min.

  RMSE  — Root Mean Squared Error: like MAE but penalises large errors more
           heavily. More sensitive to outliers than MAE. Lower is better.

  R²    — Coefficient of Determination: proportion of variance in delivery
           time explained by the model. Range: (-inf, 1.0]. 1.0 = perfect.
           > 0.7 is generally considered a useful model for this domain.

Overfitting check
-----------------
  Train vs Test MAE/RMSE are compared for each model. A large gap (train much
  better than test) indicates overfitting. The gap is printed and flagged.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV = os.path.join("data", "processed", "zomato_features.csv")
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.joblib")
META_PATH    = os.path.join(MODEL_DIR, "model_metadata.json")
COMPARE_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
TARGET       = "Time_taken (min)"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42   # fixed seed for reproducibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def evaluate(model, X_tr, y_tr, X_te, y_te, name: str) -> dict:
    """
    Evaluate a fitted model on both train and test sets.
    Returns a dict of metrics.
    """
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    mae_tr  = mean_absolute_error(y_tr, pred_tr)
    mae_te  = mean_absolute_error(y_te, pred_te)
    rmse_tr = mean_squared_error(y_tr, pred_tr) ** 0.5
    rmse_te = mean_squared_error(y_te, pred_te) ** 0.5
    r2_tr   = r2_score(y_tr, pred_tr)
    r2_te   = r2_score(y_te, pred_te)

    # Overfitting: train error is meaningfully lower than test error.
    # We flag it when test MAE exceeds train MAE by more than 1.5 minutes.
    overfit_gap = mae_te - mae_tr   # positive means test is worse than train (normal)
    overfit_flag = "YES" if overfit_gap > 1.5 else "NO"

    print(f"\n  [{name}]")
    print(f"    Train  — MAE: {mae_tr:.2f} min  RMSE: {rmse_tr:.2f} min  R²: {r2_tr:.4f}")
    print(f"    Test   — MAE: {mae_te:.2f} min  RMSE: {rmse_te:.2f} min  R²: {r2_te:.4f}")
    print(f"    Overfit check: train MAE={mae_tr:.2f} vs test MAE={mae_te:.2f}  "
          f"gap={overfit_gap:.2f}  OVERFIT={overfit_flag}")

    return {
        "model":      name,
        "train_mae":  round(mae_tr,  3),
        "test_mae":   round(mae_te,  3),
        "train_rmse": round(rmse_tr, 3),
        "test_rmse":  round(rmse_te, 3),
        "train_r2":   round(r2_tr,   4),
        "test_r2":    round(r2_te,   4),
        "overfit":    overfit_flag,
    }


# ---------------------------------------------------------------------------
# STEP 1 — Load and split
# ---------------------------------------------------------------------------
print_section("STEP 1: Loading feature dataset")

df = pd.read_csv(FEATURES_CSV)
print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

X = df.drop(columns=[TARGET])
y = df[TARGET]

FEATURE_NAMES = X.columns.tolist()
print(f"  Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
print(f"  Target: {TARGET}  |  range=[{y.min()}, {y.max()}]  mean={y.mean():.2f}")

# 80 / 20 split, stratified by rounded target to keep class balance
# We use a plain random split because delivery time is continuous.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"\n  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

# ---------------------------------------------------------------------------
# STEP 2 — Define models
#
# Linear Regression and Ridge are wrapped in a Pipeline with StandardScaler
# because linear models are sensitive to feature scale differences.
#
# Tree-based models (Decision Tree, Random Forest, Gradient Boosting) do NOT
# need scaling — they split on individual feature values, so scale is irrelevant.
# ---------------------------------------------------------------------------
print_section("STEP 2: Defining models")

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,          # limit depth to prevent extreme overfitting
        min_samples_leaf=20,   # a leaf must have at least 20 samples
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,      # 200 trees — good balance of accuracy and speed
        max_depth=15,          # moderate depth per tree
        min_samples_leaf=10,
        n_jobs=-1,             # use all CPU cores
        random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,    # small learning rate — more trees, less overfit
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,         # use 80% of data per tree (reduces variance)
        random_state=RANDOM_STATE
    ),
}

for name in models:
    print(f"  Registered: {name}")

# ---------------------------------------------------------------------------
# STEP 3 — Train and evaluate all models
# ---------------------------------------------------------------------------
print_section("STEP 3: Training and evaluating all models")
print("\n  Metric explanation:")
print("  MAE  = Mean Absolute Error (avg error in minutes — lower is better)")
print("  RMSE = Root Mean Squared Error (penalises large errors — lower is better)")
print("  R²   = Proportion of variance explained (higher is better, max=1.0)")

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\n  Training: {name} ...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
    metrics = evaluate(model, X_train, y_train, X_test, y_test, name)
    results.append(metrics)

# ---------------------------------------------------------------------------
# STEP 4 — Cross-validation on the best candidate
#
# Cross-validation trains the model on k different train/test splits and
# averages the results. This gives a more reliable performance estimate
# than a single split, and also checks for overfitting.
# We run it only on the two best tree models to save time.
# ---------------------------------------------------------------------------
print_section("STEP 4: 5-fold cross-validation (Random Forest & Gradient Boosting)")

for name in ["Random Forest", "Gradient Boosting"]:
    cv_mae = -cross_val_score(
        fitted_models[name], X, y,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1
    )
    print(f"\n  {name} — 5-fold CV MAE:")
    print(f"    Folds: {[round(v, 2) for v in cv_mae]}")
    print(f"    Mean={cv_mae.mean():.2f}  Std={cv_mae.std():.2f}")

# ---------------------------------------------------------------------------
# STEP 5 — Compare and select best model
# ---------------------------------------------------------------------------
print_section("STEP 5: Model comparison")

comparison_df = pd.DataFrame(results).sort_values("test_mae")
print(comparison_df.to_string(index=False))

# Best model = lowest test MAE
best_row   = comparison_df.iloc[0]
best_name  = best_row["model"]
best_model = fitted_models[best_name]

print(f"\n  Best model: {best_name}")
print(f"    Test MAE  = {best_row['test_mae']} min")
print(f"    Test RMSE = {best_row['test_rmse']} min")
print(f"    Test R²   = {best_row['test_r2']}")

# ---------------------------------------------------------------------------
# STEP 6 — Feature importance (tree-based models only)
# ---------------------------------------------------------------------------
print_section("STEP 6: Feature importance (best model)")

try:
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        importances = best_model.named_steps.get("model").coef_
        importances = np.abs(importances)    # use absolute value for linear models
    else:
        importances = None

    if importances is not None:
        imp_df = pd.DataFrame({
            "feature":    FEATURE_NAMES,
            "importance": importances
        }).sort_values("importance", ascending=False)
        print(imp_df.to_string(index=False))
except Exception as e:
    print(f"  Could not extract feature importances: {e}")

# ---------------------------------------------------------------------------
# STEP 7 — Save best model and metadata
# ---------------------------------------------------------------------------
print_section("STEP 7: Saving best model")

joblib.dump(best_model, MODEL_PATH)
print(f"  Model saved to: {MODEL_PATH}")

# Save comparison table
comparison_df.to_csv(COMPARE_PATH, index=False)
print(f"  Comparison table saved to: {COMPARE_PATH}")

# Save metadata (used by the prediction API in app.py)
metadata = {
    "best_model_name":  best_name,
    "model_path":       MODEL_PATH,
    "feature_names":    FEATURE_NAMES,
    "target":           TARGET,
    "test_mae":         float(best_row["test_mae"]),
    "test_rmse":        float(best_row["test_rmse"]),
    "test_r2":          float(best_row["test_r2"]),
    "train_rows":       len(X_train),
    "test_rows":        len(X_test),
    "random_state":     RANDOM_STATE,
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Metadata saved to: {META_PATH}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print_section("TRAINING COMPLETE")
print(f"  Best model : {best_name}")
print(f"  Test MAE   : {best_row['test_mae']} min  (on average, predictions are off by this many minutes)")
print(f"  Test RMSE  : {best_row['test_rmse']} min  (larger errors are penalised more heavily)")
print(f"  Test R²    : {best_row['test_r2']}   (proportion of delivery-time variance explained)")
print(f"  Overfit    : {best_row['overfit']}")
print(f"\n  Model file : {MODEL_PATH}")
