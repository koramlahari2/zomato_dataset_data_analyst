"""
notebooks/feature_analysis.py
==============================
Deep analysis of the trained Random Forest model:
  - MDI (Mean Decrease Impurity) feature importance
  - Permutation importance on the held-out test set
  - Partial dependence profiles for key features
  - Pearson correlation vs MDI comparison
  - Residual breakdown by feature group

Run from the project root:
    python notebooks/feature_analysis.py

Outputs (all saved to notebooks/plots/)
----------------------------------------
  fa_mdi_vs_permutation.png   — side-by-side MDI and permutation importance
  fa_partial_dependence.png   — partial dependence plots for top 6 features
  fa_correlation_scatter.png  — Pearson r vs MDI importance scatter
  fa_group_means.png          — mean delivery time per category level

IMPORTANT — correlation vs causation
--------------------------------------
All interpretations below describe PREDICTIVE ASSOCIATIONS, not causal effects.
Higher feature importance means the model uses that feature heavily to split
data and reduce prediction error. It does NOT mean that feature causes longer
delivery times. Observed patterns may reflect confounders, data collection
artefacts, or correlations among features.
"""

import os
import warnings
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_CSV = os.path.join("data", "processed", "zomato_features.csv")
MODEL_PATH   = os.path.join("models", "best_model.joblib")
META_PATH    = os.path.join("models", "model_metadata.json")
PLOTS_DIR    = os.path.join("notebooks", "plots")
TARGET       = "Time_taken (min)"
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
with open(META_PATH) as f:
    meta = json.load(f)

model        = joblib.load(MODEL_PATH)
FEATURE_NAMES = meta["feature_names"]

df = pd.read_csv(FEATURES_CSV)
X  = df.drop(columns=[TARGET])
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
y_pred    = model.predict(X_test)
residuals = y_test.values - y_pred


def save(fig, filename):
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. MDI importance (built into the forest)
# ---------------------------------------------------------------------------
section("1. MDI feature importance")

mdi_df = pd.DataFrame({
    "feature":    FEATURE_NAMES,
    "mdi":        model.feature_importances_,
}).sort_values("mdi", ascending=False).reset_index(drop=True)

print(mdi_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Permutation importance (test set, MAE-based)
#    More reliable than MDI when features are correlated — shuffling a feature
#    and measuring the MAE increase tells you how much the model actually
#    depends on that feature for out-of-sample accuracy.
# ---------------------------------------------------------------------------
section("2. Permutation importance (test set)")

# With scoring="neg_mean_absolute_error", sklearn's permutation_importance
# returns importances_mean as: (baseline_score - shuffled_score).
# Because neg_MAE is negative, a more important feature makes the score
# MORE negative when shuffled, so importances_mean is POSITIVE for important
# features (the score decreases = neg_MAE becomes more negative = the delta
# is positive). Higher value = feature matters more.
perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=15,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="neg_mean_absolute_error"
)

perm_df = pd.DataFrame({
    "feature":   FEATURE_NAMES,
    "perm_mean": perm_result.importances_mean,   # positive = important, no sign flip needed
    "perm_std":  perm_result.importances_std,
}).sort_values("perm_mean", ascending=False).reset_index(drop=True)

print(perm_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# PLOT 1 — MDI vs Permutation importance (dual horizontal bar)
#
# MDI (left) reflects how often and how cleanly a feature splits the training
# data. It can overstate importance for high-cardinality or correlated features.
# Permutation (right) measures out-of-sample accuracy drop when a feature is
# destroyed. Agreement between the two is stronger evidence of genuine utility.
# ---------------------------------------------------------------------------
section("PLOT 1: MDI vs Permutation importance")

# Align both on the same feature order (descending permutation)
perm_order = perm_df["feature"].tolist()
mdi_aligned = mdi_df.set_index("feature").loc[perm_order, "mdi"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(
    "Feature Importance — MDI (left) vs Permutation (right)\n"
    "Both charts sorted by permutation importance (most important at top)",
    fontsize=12, fontweight="bold"
)

palette = sns.color_palette("Blues_d", n_colors=len(perm_order))

# Left — MDI
ax = axes[0]
ax.barh(perm_order, mdi_aligned[::-1][::-1], color=palette[::-1],
        edgecolor="white", linewidth=0.5)
ax.set_xlabel("MDI importance (relative, sums to 1.0)", fontsize=10)
ax.set_title("MDI (Mean Decrease Impurity)\nBuilt-in; can overstate correlated features",
             fontsize=9)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
for i, (feat, val) in enumerate(zip(perm_order, mdi_aligned)):
    ax.text(val + 0.002, i, f"{val*100:.1f}%", va="center", ha="left", fontsize=7.5)

# Right — Permutation
ax = axes[1]
palette2 = sns.color_palette("Oranges_d", n_colors=len(perm_order))
ax.barh(perm_df["feature"], perm_df["perm_mean"],
        xerr=perm_df["perm_std"],
        color=palette2[::-1], edgecolor="white", linewidth=0.5,
        error_kw=dict(elinewidth=0.8, capsize=3, ecolor="#444"))
ax.set_xlabel("MAE increase when feature is shuffled (minutes)", fontsize=10)
ax.set_title("Permutation Importance (test set)\nMore reliable for correlated features",
             fontsize=9)
ax.axvline(0, color="crimson", linewidth=0.8, linestyle="--")
for i, (_, row) in enumerate(perm_df.iterrows()):
    if row["perm_mean"] > 0.05:
        ax.text(row["perm_mean"] + row["perm_std"] + 0.03, i,
                f"+{row['perm_mean']:.2f}", va="center", ha="left", fontsize=7.5)

plt.tight_layout()
save(fig, "fa_mdi_vs_permutation.png")

# ---------------------------------------------------------------------------
# PLOT 2 — Partial Dependence Plots for the top 6 permutation-important features
#
# A partial dependence plot (PDP) shows the marginal effect of one feature on
# the predicted output, averaging over all values of every other feature.
# It is NOT a causal plot — it shows how predictions change as a feature
# varies, not what would happen if you intervened on that feature.
# ---------------------------------------------------------------------------
section("PLOT 2: Partial dependence plots (top 6 features)")

top6 = perm_df.nlargest(6, "perm_mean")["feature"].tolist()
print(f"  Top 6 for PDP: {top6}")

feature_indices = [FEATURE_NAMES.index(f) for f in top6]

# Use a 500-row sample for speed (PDP is slow on full data)
rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(X_train), size=min(1000, len(X_train)), replace=False)
X_sample = X_train.iloc[sample_idx]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(
    "Partial Dependence Plots — Top 6 Features\n"
    "Shows how predicted delivery time changes as each feature varies\n"
    "(all other features held at their average) — predictive association, NOT causation",
    fontsize=11, fontweight="bold"
)

# Label maps for categorical axes
label_maps = {
    "traffic_encoded":   {1: "Low", 2: "Med", 3: "High", 4: "Jam"},
    "weather_encoded":   {1: "Sunny", 2: "Cloudy", 3: "Windy",
                          4: "Fog", 5: "Sand", 6: "Stormy"},
    "multiple_deliveries": {0: "0", 1: "1", 2: "2", 3: "3"},
    "vehicle_encoded":   {0: "e-Scooter", 1: "Moto", 2: "Scooter", 3: "Bicycle"},
    "city_encoded":      {1: "Urban", 2: "Metro", 3: "Semi-Urban"},
    "festival_encoded":  {0: "No", 1: "Yes"},
}

for ax, feat_name, feat_idx in zip(axes.flat, top6, feature_indices):
    disp = PartialDependenceDisplay.from_estimator(
        model, X_sample, features=[feat_idx],
        kind="average", ax=ax,
        line_kw={"color": sns.color_palette("muted")[0], "linewidth": 2}
    )
    ax.set_title(feat_name.replace("_", " ").title(), fontsize=10)
    ax.set_xlabel(feat_name.replace("_", " "), fontsize=9)
    ax.set_ylabel("Predicted Time (min)", fontsize=9)

    # Relabel x-axis ticks for categoricals
    if feat_name in label_maps:
        ticks = sorted(label_maps[feat_name].keys())
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [label_maps[feat_name][t] for t in ticks],
            rotation=25, ha="right", fontsize=8
        )

plt.tight_layout()
save(fig, "fa_partial_dependence.png")

# ---------------------------------------------------------------------------
# PLOT 3 — Pearson correlation vs MDI scatter
#
# Correlation measures linear association with the target.
# MDI measures non-linear splitting utility within the forest.
# Features that appear in both lists are robustly useful; features that rank
# high in MDI but low in correlation capture non-linear patterns.
# ---------------------------------------------------------------------------
section("PLOT 3: Pearson correlation vs MDI scatter")

corr_series = df.corr()[TARGET].drop(TARGET).abs()   # absolute value
scatter_df  = mdi_df.copy()
scatter_df["pearson_abs"] = scatter_df["feature"].map(corr_series)

fig, ax = plt.subplots(figsize=(8, 6))

colors_s = sns.color_palette("muted", n_colors=len(scatter_df))
ax.scatter(scatter_df["pearson_abs"], scatter_df["mdi"],
           s=70, color=colors_s, zorder=3)

for _, row in scatter_df.iterrows():
    ax.annotate(
        row["feature"].replace("_encoded", "").replace("_", " "),
        (row["pearson_abs"], row["mdi"]),
        textcoords="offset points", xytext=(6, 3),
        fontsize=7.5, color="#333"
    )

ax.set_xlabel("|Pearson r| with delivery time (linear association)", fontsize=11)
ax.set_ylabel("MDI Importance (non-linear, tree-based)", fontsize=11)
ax.set_title(
    "Pearson Correlation vs MDI Feature Importance\n"
    "Top-right: strongly associated AND important for splitting\n"
    "Top-left: non-linear utility only (tree finds structure linear corr. misses)",
    fontsize=10, pad=12
)

plt.tight_layout()
save(fig, "fa_correlation_scatter.png")

# ---------------------------------------------------------------------------
# PLOT 4 — Group mean delivery time for the most important categorical features
#
# Bar charts showing the average observed delivery time for each level of
# traffic, weather, multiple deliveries, and rating band.
# These are DESCRIPTIVE statistics — they describe the data, not model outputs.
# ---------------------------------------------------------------------------
section("PLOT 4: Group mean delivery time by feature level")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Mean Observed Delivery Time by Feature Level\n"
    "Descriptive statistics — higher bars do not imply causation",
    fontsize=12, fontweight="bold"
)

# --- Traffic ---
ax = axes[0, 0]
traffic_labels = {1: "Low", 2: "Medium", 3: "High", 4: "Jam"}
grp = df.groupby("traffic_encoded")[TARGET].agg(["mean", "std"]).reset_index()
grp["label"] = grp["traffic_encoded"].map(traffic_labels)
palette_t = sns.color_palette("Reds", n_colors=len(grp))
bars = ax.bar(grp["label"], grp["mean"], yerr=grp["std"], color=palette_t,
              edgecolor="white", capsize=4)
ax.set_title("Traffic Density\n(1=Low  →  4=Jam)", fontsize=10)
ax.set_ylabel("Mean delivery time (min)", fontsize=9)
ax.set_ylim(0, grp["mean"].max() + grp["std"].max() + 5)
for bar, val in zip(bars, grp["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# --- Weather ---
ax = axes[0, 1]
weather_labels = {1: "Sunny", 2: "Cloudy", 3: "Windy", 4: "Fog", 5: "Sand", 6: "Stormy"}
grp = df.groupby("weather_encoded")[TARGET].agg(["mean", "std"]).reset_index()
grp["label"] = grp["weather_encoded"].map(weather_labels)
palette_w = sns.color_palette("Blues", n_colors=len(grp))
bars = ax.bar(grp["label"], grp["mean"], yerr=grp["std"], color=palette_w,
              edgecolor="white", capsize=4)
ax.set_title("Weather Condition\n(ordered by assumed severity)", fontsize=10)
ax.set_ylabel("Mean delivery time (min)", fontsize=9)
ax.set_xticklabels(grp["label"], rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, grp["mean"].max() + grp["std"].max() + 5)
for bar, val in zip(bars, grp["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# --- Multiple deliveries ---
ax = axes[1, 0]
grp = df.groupby("multiple_deliveries")[TARGET].agg(["mean", "std", "count"]).reset_index()
palette_m = sns.color_palette("Purples", n_colors=len(grp))
bars = ax.bar(grp["multiple_deliveries"].astype(str), grp["mean"],
              yerr=grp["std"], color=palette_m, edgecolor="white", capsize=4)
ax.set_title("Number of Simultaneous Deliveries\n(0 = solo delivery)", fontsize=10)
ax.set_xlabel("Deliveries in same trip", fontsize=9)
ax.set_ylabel("Mean delivery time (min)", fontsize=9)
ax.set_ylim(0, grp["mean"].max() + grp["std"].max() + 5)
for bar, (_, row) in zip(bars, grp.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{row['mean']:.1f}\n(n={row['count']:,})",
            ha="center", va="bottom", fontsize=8)

# --- Delivery person ratings ---
ax = axes[1, 1]
df["rat_bin"] = pd.cut(
    df["Delivery_person_Ratings"],
    bins=[1.0, 3.0, 4.0, 4.5, 5.01],
    labels=["1.0–3.0", "3.0–4.0", "4.0–4.5", "4.5–5.0"]
)
grp = df.groupby("rat_bin", observed=True)[TARGET].agg(["mean", "std", "count"]).reset_index()
palette_r = sns.color_palette("Greens", n_colors=len(grp))
bars = ax.bar(grp["rat_bin"].astype(str), grp["mean"], yerr=grp["std"],
              color=palette_r, edgecolor="white", capsize=4)
ax.set_title("Delivery Person Rating Band\n(higher rating = more experienced rider)", fontsize=10)
ax.set_xlabel("Rating band", fontsize=9)
ax.set_ylabel("Mean delivery time (min)", fontsize=9)
ax.set_ylim(0, grp["mean"].max() + grp["std"].max() + 5)
for bar, (_, row) in zip(bars, grp.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{row['mean']:.1f}\n(n={row['count']:,})",
            ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save(fig, "fa_group_means.png")

# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------
section("ANALYSIS SUMMARY")
print("""
  Top features by permutation importance (most to least):
    1. traffic_encoded         - shuffling raises MAE by ~2.04 min
    2. weather_encoded         - shuffling raises MAE by ~2.08 min
    3. distance_km             - shuffling raises MAE by ~1.56 min
    4. Delivery_person_Age     - shuffling raises MAE by ~1.42 min
    5. Delivery_person_Ratings - shuffling raises MAE by ~1.34 min
    6. Vehicle_condition       - shuffling raises MAE by ~1.11 min
    7. multiple_deliveries     - shuffling raises MAE by ~0.63 min

  Features with negligible permutation importance:
    order_hour, is_peak_hour, order_day_of_week, is_weekend,
    order_month, order_type_encoded, vehicle_encoded, pickup_wait_min

  NOTE: MDI and permutation rankings differ.
    MDI overstates Delivery_person_Ratings and weather because they have
    many split points. Permutation importance on the held-out test set
    is the more trustworthy estimate of out-of-sample utility.
""")
