"""
notebooks/model_evaluation.py
==============================
Visual evaluation of the trained food-delivery-time models.

Run from the project root:
    python notebooks/model_evaluation.py

Inputs
------
  data/processed/zomato_features.csv  — feature-engineered dataset
  models/best_model.joblib            — saved best model (Random Forest)
  models/model_comparison.csv         — comparison table from training

Outputs (saved to notebooks/plots/)
-------------------------------------
  model_comparison_bar.png   — MAE / RMSE / R² bar chart for all 5 models
  actual_vs_predicted.png    — scatter of true vs predicted delivery time
  residuals_distribution.png — histogram of prediction errors
  feature_importance.png     — top feature importances from the best model

What each plot shows
--------------------
  1. Model comparison bar    — side-by-side test metrics let you see which
                               model is best and by how much.
  2. Actual vs predicted     — ideal predictions lie on the y=x diagonal.
                               Scatter around it shows where the model errs.
  3. Residuals distribution  — residuals (actual - predicted) should be
                               roughly centred on zero. A skewed or bimodal
                               histogram signals systematic bias.
  4. Feature importance      — which input features drive the model's
                               predictions most. High-importance features are
                               the key drivers of delivery time.
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
FEATURES_CSV  = os.path.join("data", "processed", "zomato_features.csv")
MODEL_PATH    = os.path.join("models", "best_model.joblib")
COMPARE_PATH  = os.path.join("models", "model_comparison.csv")
PLOTS_DIR     = os.path.join("notebooks", "plots")
TARGET        = "Time_taken (min)"
RANDOM_STATE  = 42

os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent style for all plots
sns.set_theme(style="whitegrid", palette="muted")
TITLE_PAD = 14

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# STEP 1 — Load data and reproduce the same train / test split
#
# We must use the same random_state=42 and test_size=0.20 as in train.py so
# the test set here is identical to the one used during model training.
# Evaluating on a different split would give misleading results.
# ---------------------------------------------------------------------------
print_section("STEP 1: Loading data and recreating train/test split")

df = pd.read_csv(FEATURES_CSV)
X  = df.drop(columns=[TARGET])
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"  Dataset: {df.shape[0]:,} rows  |  "
      f"Train: {len(X_train):,}  Test: {len(X_test):,}")

# ---------------------------------------------------------------------------
# STEP 2 — Load the saved best model and generate test predictions
# ---------------------------------------------------------------------------
print_section("STEP 2: Loading best model and generating predictions")

best_model = joblib.load(MODEL_PATH)
print(f"  Loaded model from: {MODEL_PATH}")
print(f"  Model type: {type(best_model).__name__}")

y_pred = best_model.predict(X_test)
residuals = y_test.values - y_pred

mae  = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))
print(f"\n  Test MAE  = {mae:.3f} min")
print(f"  Test RMSE = {rmse:.3f} min")
print(f"  Residuals — mean={residuals.mean():.3f}  std={residuals.std():.3f}  "
      f"min={residuals.min():.1f}  max={residuals.max():.1f}")

# ---------------------------------------------------------------------------
# PLOT 1 — Model comparison bar chart
#
# Shows test MAE, RMSE, and R² for all five models side-by-side.
# Makes it easy to see which model performs best on each metric.
# ---------------------------------------------------------------------------
print_section("PLOT 1: Model comparison bar chart")

compare_df = pd.read_csv(COMPARE_PATH)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold",
             y=1.02)

metrics = [
    ("test_mae",  "Test MAE (minutes)",              "lower is better"),
    ("test_rmse", "Test RMSE (minutes)",             "lower is better"),
    ("test_r2",   "Test R² (proportion explained)",  "higher is better"),
]

colors = sns.color_palette("muted", n_colors=len(compare_df))

for ax, (col, ylabel, note) in zip(axes, metrics):
    bars = ax.bar(compare_df["model"], compare_df[col], color=colors, edgecolor="white",
                  linewidth=0.8)
    ax.set_title(f"{ylabel}\n({note})", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticklabels(compare_df["model"], rotation=25, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Annotate bar values
    for bar, val in zip(bars, compare_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save(fig, "model_comparison_bar.png")

# ---------------------------------------------------------------------------
# PLOT 2 — Actual vs Predicted scatter
#
# Ideal predictions lie exactly on the y=x diagonal.  Points above the line
# mean the model under-predicted (actual was longer than predicted).  Points
# below mean it over-predicted.  The tighter the cloud around the diagonal,
# the better the model.
# ---------------------------------------------------------------------------
print_section("PLOT 2: Actual vs Predicted delivery time")

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_test, y_pred, alpha=0.25, s=12, color=sns.color_palette("muted")[0],
           label="Predictions", rasterized=True)

# Perfect-prediction reference line
lo = min(y_test.min(), y_pred.min()) - 2
hi = max(y_test.max(), y_pred.max()) + 2
ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.5, linestyle="--",
        label="Perfect prediction (y = x)")

ax.set_xlabel("Actual Delivery Time (minutes)", fontsize=11)
ax.set_ylabel("Predicted Delivery Time (minutes)", fontsize=11)
ax.set_title(
    "Actual vs Predicted Delivery Time\n"
    "Points close to the red line = accurate predictions",
    fontsize=12, pad=TITLE_PAD
)
ax.legend(fontsize=9)

# Annotate with test metrics
ax.text(0.03, 0.96,
        f"MAE  = {mae:.2f} min\nRMSE = {rmse:.2f} min",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()
save(fig, "actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# PLOT 3 — Residuals distribution
#
# Residual = actual − predicted.
# A good model's residuals are:
#   • Centred near zero  (no systematic bias)
#   • Roughly bell-shaped  (normally distributed errors)
# A skewed distribution means the model consistently under- or over-estimates
# for certain ranges of delivery time.
# ---------------------------------------------------------------------------
print_section("PLOT 3: Residuals distribution")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Residual Analysis (Actual − Predicted)", fontsize=13,
             fontweight="bold")

# — Histogram
ax = axes[0]
ax.hist(residuals, bins=50, color=sns.color_palette("muted")[1],
        edgecolor="white", linewidth=0.5)
ax.axvline(0, color="crimson", linewidth=1.5, linestyle="--", label="Zero error")
ax.axvline(residuals.mean(), color="navy", linewidth=1.2, linestyle=":",
           label=f"Mean = {residuals.mean():.2f} min")
ax.set_xlabel("Residual (minutes)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(
    "Residual Histogram\nShould be centred on 0 and roughly bell-shaped",
    fontsize=10, pad=TITLE_PAD
)
ax.legend(fontsize=9)

# — Residuals vs Predicted (heteroscedasticity check)
# A flat horizontal band around zero is ideal.
# A funnel shape (spread increases with predicted value) indicates
# heteroscedasticity — the model is less reliable for longer delivery times.
ax = axes[1]
ax.scatter(y_pred, residuals, alpha=0.2, s=10,
           color=sns.color_palette("muted")[2], rasterized=True)
ax.axhline(0, color="crimson", linewidth=1.5, linestyle="--")
ax.set_xlabel("Predicted Delivery Time (minutes)", fontsize=11)
ax.set_ylabel("Residual (minutes)", fontsize=11)
ax.set_title(
    "Residuals vs Predicted Values\nFlat band around 0 = consistent accuracy",
    fontsize=10, pad=TITLE_PAD
)

plt.tight_layout()
save(fig, "residuals_distribution.png")

# ---------------------------------------------------------------------------
# PLOT 4 — Feature importance
#
# Feature importance (for Random Forest / Gradient Boosting) measures how
# much each feature reduces prediction error on average across all trees.
# Higher importance = that feature is more useful for predicting delivery time.
#
# Important caveat: importance values sum to 1.0, so they represent relative
# contribution, not absolute effect size.
# ---------------------------------------------------------------------------
print_section("PLOT 4: Feature importance")

FEATURE_NAMES = X.columns.tolist()

try:
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        inner = best_model.named_steps.get("model")
        importances = np.abs(inner.coef_) if hasattr(inner, "coef_") else None
    else:
        importances = None

    if importances is not None:
        imp_df = (
            pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances})
            .sort_values("importance", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(8, 7))

        colors_imp = sns.color_palette("Blues_d", n_colors=len(imp_df))
        bars = ax.barh(imp_df["feature"], imp_df["importance"],
                       color=colors_imp, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Feature Importance (relative contribution)", fontsize=11)
        ax.set_title(
            f"Feature Importance — {type(best_model).__name__}\n"
            "Higher = stronger driver of predicted delivery time",
            fontsize=11, pad=TITLE_PAD
        )
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        # Annotate percentage values
        for bar, val in zip(bars, imp_df["importance"]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", ha="left", fontsize=8)

        plt.tight_layout()
        save(fig, "feature_importance.png")
    else:
        print("  Model does not expose feature importances — skipping plot.")
except Exception as e:
    print(f"  Could not plot feature importances: {e}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print_section("EVALUATION COMPLETE")
print(f"  Best model : {type(best_model).__name__}")
print(f"  Test MAE   : {mae:.3f} min")
print(f"               i.e. on average, predictions are off by {mae:.1f} minutes.")
print(f"  Test RMSE  : {rmse:.3f} min")
print(f"               i.e. large errors are penalised more than small ones.")
print(f"  Residuals  : mean={residuals.mean():.3f} (close to 0 = unbiased)")
print(f"\n  All plots saved to: {PLOTS_DIR}/")
print(f"    model_comparison_bar.png")
print(f"    actual_vs_predicted.png")
print(f"    residuals_distribution.png")
print(f"    feature_importance.png")
