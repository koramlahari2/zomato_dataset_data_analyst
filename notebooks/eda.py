"""
notebooks/eda.py
================
Exploratory Data Analysis for the Zomato Food Delivery dataset.

Run from the project root:
    python notebooks/eda.py

All plots are saved to:  notebooks/plots/

Analyses covered:
  1.  Overall delivery-time distribution
  2.  Delivery time by weather condition
  3.  Delivery time by traffic density
  4.  Delivery time by vehicle type
  5.  Delivery time by vehicle condition
  6.  Delivery time by order type
  7.  Delivery time by city
  8.  Delivery time by delivery-person rating
  9.  Delivery time by number of multiple deliveries
  10. Delivery time vs. Haversine delivery distance
"""

import os
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLEANED_CSV  = os.path.join("data", "processed", "zomato_cleaned.csv")
PLOTS_DIR    = os.path.join("notebooks", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent visual style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = "Blues_d"
TARGET    = "Time_taken (min)"
BLUE      = "#3b82d4"
ACCENT    = "#e05c2e"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, filename: str) -> None:
    """Save figure to the plots directory and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate straight-line distance between two GPS points in kilometres.
    Uses the Haversine formula — accurate enough for city-level distances.
    """
    R = 6371.0                              # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading cleaned dataset ...")
df = pd.read_csv(CLEANED_CSV)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Target range: {df[TARGET].min()}–{df[TARGET].max()} min  |  mean={df[TARGET].mean():.1f}")

# Ordinal order for traffic density
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]
df["Road_traffic_density"] = pd.Categorical(
    df["Road_traffic_density"], categories=TRAFFIC_ORDER, ordered=True
)

# ============================================================
# 1. Overall delivery-time distribution
# ============================================================
section("1. Overall delivery-time distribution")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Overall Delivery Time Distribution", fontsize=14, fontweight="bold")

# Histogram + KDE
axes[0].hist(df[TARGET], bins=30, color=BLUE, edgecolor="white", alpha=0.85, density=True)
df[TARGET].plot.kde(ax=axes[0], color=ACCENT, linewidth=2)
axes[0].set_xlabel("Delivery Time (min)")
axes[0].set_ylabel("Density")
axes[0].set_title("Histogram with KDE")
mean_t = df[TARGET].mean()
median_t = df[TARGET].median()
axes[0].axvline(mean_t,   color="red",    linestyle="--", linewidth=1.4, label=f"Mean  {mean_t:.1f}")
axes[0].axvline(median_t, color="green",  linestyle=":",  linewidth=1.4, label=f"Median {median_t:.1f}")
axes[0].legend(fontsize=9)

# Box plot
axes[1].boxplot(df[TARGET].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor=BLUE, color="navy"),
                medianprops=dict(color=ACCENT, linewidth=2),
                whiskerprops=dict(color="navy"),
                capprops=dict(color="navy"),
                flierprops=dict(marker="o", color="grey", alpha=0.3, markersize=3))
axes[1].set_ylabel("Delivery Time (min)")
axes[1].set_title("Box Plot")
axes[1].set_xticks([])

plt.tight_layout()
save(fig, "01_delivery_time_distribution.png")

# Stats print
q1, q3 = df[TARGET].quantile([0.25, 0.75])
print(f"  Mean={mean_t:.1f}  Median={median_t:.1f}  Std={df[TARGET].std():.1f}")
print(f"  Q1={q1}  Q3={q3}  IQR={q3-q1}")
print(f"  Min={df[TARGET].min()}  Max={df[TARGET].max()}")
print("  INSIGHT: Delivery times are roughly uniformly spread between 10 and 54 min with a")
print("  slight peak around 20-30 min. The mean (26) and median (25) are very close,")
print("  indicating no extreme skew — the distribution is near-symmetric.")

# ============================================================
# 2. Delivery time by weather condition
# ============================================================
section("2. Delivery time by weather condition")

weather_stats = (
    df.groupby("Weather_conditions")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
    .reset_index()
)
print(weather_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Delivery Time by Weather Condition", fontsize=14, fontweight="bold")

order_w = weather_stats["Weather_conditions"].tolist()

sns.boxplot(data=df, x="Weather_conditions", y=TARGET,
            order=order_w, palette="Blues", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Weather")
axes[0].set_ylabel("Delivery Time (min)")
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(data=weather_stats, x="Weather_conditions", y="Mean",
            order=order_w, palette="Blues_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Weather")
axes[1].set_ylabel("Mean Time (min)")
axes[1].tick_params(axis="x", rotation=20)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "02_delivery_time_by_weather.png")
print("  INSIGHT: All weather types produce very similar mean delivery times (roughly 26 min).")
print("  No single weather condition dramatically inflates time on its own — suggesting")
print("  traffic density may be a stronger driver than weather alone.")

# ============================================================
# 3. Delivery time by traffic density
# ============================================================
section("3. Delivery time by traffic density")

traffic_stats = (
    df.groupby("Road_traffic_density", observed=True)[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .reset_index()
)
print(traffic_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Delivery Time by Road Traffic Density", fontsize=14, fontweight="bold")

sns.boxplot(data=df, x="Road_traffic_density", y=TARGET,
            order=TRAFFIC_ORDER, palette="Reds", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Traffic Density")
axes[0].set_ylabel("Delivery Time (min)")

sns.barplot(data=traffic_stats, x="Road_traffic_density", y="Mean",
            order=TRAFFIC_ORDER, palette="Reds_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Traffic Density")
axes[1].set_ylabel("Mean Time (min)")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "03_delivery_time_by_traffic.png")
print("  INSIGHT: Traffic density shows a clear increasing trend from Low to Jam.")
print("  'Jam' conditions add roughly 8-10 extra minutes vs 'Low' traffic.")
print("  This makes traffic density one of the most actionable predictors of delay.")

# ============================================================
# 4. Delivery time by vehicle type
# ============================================================
section("4. Delivery time by vehicle type")

vehicle_stats = (
    df.groupby("Type_of_vehicle")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
    .reset_index()
)
print(vehicle_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Delivery Time by Vehicle Type", fontsize=14, fontweight="bold")

order_v = vehicle_stats["Type_of_vehicle"].tolist()
sns.boxplot(data=df, x="Type_of_vehicle", y=TARGET,
            order=order_v, palette="Greens", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Vehicle Type")
axes[0].set_ylabel("Delivery Time (min)")
axes[0].tick_params(axis="x", rotation=15)

sns.barplot(data=vehicle_stats, x="Type_of_vehicle", y="Mean",
            order=order_v, palette="Greens_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Vehicle Type")
axes[1].set_ylabel("Mean Time (min)")
axes[1].tick_params(axis="x", rotation=15)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "04_delivery_time_by_vehicle_type.png")
print("  INSIGHT: Motorcycles have the highest mean delivery time (~27.6 min), while")
print("  scooters and electric scooters are meaningfully faster (~24.5 min each).")
print("  Bicycles appear in only 68 orders, making that group statistically unreliable.")
print("  Scooters and e-scooters may be favoured on shorter, dense city routes.")

# ============================================================
# 5. Delivery time by vehicle condition
# ============================================================
section("5. Delivery time by vehicle condition")

cond_stats = (
    df.groupby("Vehicle_condition")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .reset_index()
)
print(cond_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Delivery Time by Vehicle Condition (0=worst, 3=best)", fontsize=14, fontweight="bold")

sns.boxplot(data=df, x="Vehicle_condition", y=TARGET,
            palette="Purples", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Vehicle Condition Score")
axes[0].set_ylabel("Delivery Time (min)")

sns.barplot(data=cond_stats, x="Vehicle_condition", y="Mean",
            palette="Purples_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Vehicle Condition Score")
axes[1].set_ylabel("Mean Time (min)")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "05_delivery_time_by_vehicle_condition.png")
print("  INSIGHT: Condition 0 (worst) has a notably higher mean (~30 min) vs.")
print("  conditions 1 and 2 (~24 min each). Poorly maintained vehicles do appear")
print("  to slow deliveries. Condition 3 has only 520 records, making it less reliable.")

# ============================================================
# 6. Delivery time by order type
# ============================================================
section("6. Delivery time by order type")

order_stats = (
    df.groupby("Type_of_order")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
    .reset_index()
)
print(order_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Delivery Time by Order Type", fontsize=14, fontweight="bold")

order_o = order_stats["Type_of_order"].tolist()
sns.boxplot(data=df, x="Type_of_order", y=TARGET,
            order=order_o, palette="Oranges", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Order Type")
axes[0].set_ylabel("Delivery Time (min)")

sns.barplot(data=order_stats, x="Type_of_order", y="Mean",
            order=order_o, palette="Oranges_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Order Type")
axes[1].set_ylabel("Mean Time (min)")
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "06_delivery_time_by_order_type.png")
print("  INSIGHT: Mean delivery time is broadly similar across Snack, Meal, Drinks,")
print("  and Buffet. Small differences may reflect kitchen prep differences (e.g. Buffet")
print("  may require more assembly time before the rider can pick up), but the effect")
print("  is not dramatic in this data.")

# ============================================================
# 7. Delivery time by city
# ============================================================
section("7. Delivery time by city")

city_stats = (
    df.groupby("City")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .sort_values("Mean", ascending=False)
    .reset_index()
)
print(city_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Delivery Time by City Tier", fontsize=14, fontweight="bold")

order_c = city_stats["City"].tolist()
sns.boxplot(data=df, x="City", y=TARGET,
            order=order_c, palette="YlOrRd", ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("City Tier")
axes[0].set_ylabel("Delivery Time (min)")

bars = sns.barplot(data=city_stats, x="City", y="Mean",
                   order=order_c, palette="YlOrRd", ax=axes[1])
axes[1].set_title("Mean & Count")
axes[1].set_xlabel("City Tier")
axes[1].set_ylabel("Mean Time (min)")
for bar, (_, row) in zip(axes[1].patches, city_stats.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}\n(n={int(row['Count']):,})",
                 ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save(fig, "07_delivery_time_by_city.png")
print("  INSIGHT: Semi-Urban areas show the highest mean delivery time — likely because")
print("  riders cover larger geographic distances. Metropolitan cities have the most")
print("  orders and a moderate average time. Urban sits in the middle.")
print("  Note: Semi-Urban has very few orders (164) so its average is less reliable.")

# ============================================================
# 8. Delivery time by delivery-person rating
# ============================================================
section("8. Delivery time by delivery-person rating")

# Round ratings to 1 decimal for grouping
df["Rating_rounded"] = df["Delivery_person_Ratings"].round(1)
rating_stats = (
    df.groupby("Rating_rounded")[TARGET]
    .agg(Mean="mean", Count="count")
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Delivery Time by Rider Rating", fontsize=14, fontweight="bold")

# Scatter: each rating bucket mean
axes[0].scatter(rating_stats["Rating_rounded"], rating_stats["Mean"],
                s=rating_stats["Count"] / 20, color=BLUE, alpha=0.75, edgecolors="navy")
axes[0].set_xlabel("Delivery Person Rating")
axes[0].set_ylabel("Mean Delivery Time (min)")
axes[0].set_title("Mean Time per Rating (bubble size = order count)")

# KDE by high vs low rating groups
mask_high = df["Delivery_person_Ratings"] >= 4.5
mask_low  = df["Delivery_person_Ratings"] <  4.5
axes[1].hist(df.loc[mask_high, TARGET], bins=25, alpha=0.55,
             color=BLUE,   label=f"Rating >= 4.5 (n={mask_high.sum():,})", density=True)
axes[1].hist(df.loc[mask_low,  TARGET], bins=25, alpha=0.55,
             color=ACCENT, label=f"Rating <  4.5 (n={mask_low.sum():,})",  density=True)
axes[1].set_xlabel("Delivery Time (min)")
axes[1].set_ylabel("Density")
axes[1].set_title("High vs. Low Rated Riders")
axes[1].legend(fontsize=9)

plt.tight_layout()
save(fig, "08_delivery_time_by_rating.png")

corr = df["Delivery_person_Ratings"].corr(df[TARGET])
print(f"  Pearson correlation (rating vs time): {corr:.4f}")
print("  INSIGHT: The Pearson correlation is -0.33 — a moderate negative relationship.")
print("  Higher-rated riders do tend to deliver somewhat faster on average, but the")
print("  signal is not strong enough to use rating alone as a reliable predictor.")

# ============================================================
# 9. Delivery time by multiple deliveries
# ============================================================
section("9. Delivery time by number of multiple deliveries")

multi_stats = (
    df.groupby("multiple_deliveries")[TARGET]
    .agg(Mean="mean", Median="median", Count="count")
    .reset_index()
    .sort_values("multiple_deliveries")
)
print(multi_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Delivery Time by Number of Simultaneous Deliveries", fontsize=14, fontweight="bold")

sns.boxplot(data=df, x="multiple_deliveries", y=TARGET,
            palette="Blues", ax=axes[0], order=[0, 1, 2, 3])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Number of Simultaneous Deliveries")
axes[0].set_ylabel("Delivery Time (min)")

sns.barplot(data=multi_stats, x="multiple_deliveries", y="Mean",
            palette="Blues_d", ax=axes[1], order=[0, 1, 2, 3])
axes[1].set_title("Mean Delivery Time")
axes[1].set_xlabel("Number of Simultaneous Deliveries")
axes[1].set_ylabel("Mean Time (min)")
for bar, (_, row) in zip(axes[1].patches, multi_stats.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=9)

plt.tight_layout()
save(fig, "09_delivery_time_by_multiple_deliveries.png")
print("  INSIGHT: Mean delivery time increases noticeably as the number of simultaneous")
print("  deliveries rises from 0 to 3. Riders handling 3 orders take significantly longer")
print("  than those with a single order. This is one of the clearest operational levers")
print("  — limiting concurrent orders would directly reduce delivery times.")

# ============================================================
# 10. Delivery time vs. delivery distance (Haversine)
# ============================================================
section("10. Delivery time vs. Haversine distance")

# Calculate distance for each row where all 4 coordinates are available
coord_cols = ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]

df_dist = df.dropna(subset=coord_cols).copy()

df_dist["distance_km"] = df_dist.apply(
    lambda r: haversine_km(
        r["Restaurant_latitude"],  r["Restaurant_longitude"],
        r["Delivery_location_latitude"], r["Delivery_location_longitude"]
    ),
    axis=1
)

# Remove implausible distances (< 0.1 km or > 25 km for city delivery)
df_dist = df_dist[(df_dist["distance_km"] >= 0.1) & (df_dist["distance_km"] <= 25)]
print(f"  Rows with valid distance: {len(df_dist):,}")
print(f"  Distance range: {df_dist['distance_km'].min():.2f} – {df_dist['distance_km'].max():.2f} km")
print(f"  Mean distance: {df_dist['distance_km'].mean():.2f} km")

corr_dist = df_dist["distance_km"].corr(df_dist[TARGET])
print(f"  Pearson correlation (distance vs time): {corr_dist:.4f}")

# Bin distance for grouped bar chart
bins  = [0, 2, 4, 6, 8, 10, 15, 25]
labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-25"]
df_dist["dist_bin"] = pd.cut(df_dist["distance_km"], bins=bins, labels=labels)
bin_stats = (
    df_dist.groupby("dist_bin", observed=True)[TARGET]
    .agg(Mean="mean", Count="count")
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Delivery Time vs. Distance (Haversine)", fontsize=14, fontweight="bold")

# Scatter with regression line (sample 3000 points for readability)
sample = df_dist.sample(min(3000, len(df_dist)), random_state=42)
axes[0].scatter(sample["distance_km"], sample[TARGET],
                alpha=0.15, s=8, color=BLUE)
# Fit a linear trend
m, b = np.polyfit(df_dist["distance_km"], df_dist[TARGET], 1)
x_line = np.linspace(df_dist["distance_km"].min(), df_dist["distance_km"].max(), 100)
axes[0].plot(x_line, m * x_line + b, color=ACCENT, linewidth=2,
             label=f"Trend  r={corr_dist:.2f}")
axes[0].set_xlabel("Delivery Distance (km)")
axes[0].set_ylabel("Delivery Time (min)")
axes[0].set_title("Scatter: Distance vs. Time (3k sample)")
axes[0].legend(fontsize=9)

# Grouped mean by distance bin
sns.barplot(data=bin_stats, x="dist_bin", y="Mean",
            palette="Blues_d", ax=axes[1])
axes[1].set_title("Mean Delivery Time by Distance Bin")
axes[1].set_xlabel("Distance (km)")
axes[1].set_ylabel("Mean Time (min)")
for bar, (_, row) in zip(axes[1].patches, bin_stats.iterrows()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom", fontsize=8)

plt.tight_layout()
save(fig, "10_delivery_time_vs_distance.png")
print("  INSIGHT: There is a positive correlation between distance and delivery time,")
print("  as expected — longer routes take more time. However, the correlation is moderate")
print("  rather than perfect, confirming that traffic, weather, and simultaneous orders")
print("  also play significant roles independent of distance.")

# ============================================================
# BONUS: Heatmap — Traffic x Weather mean delivery time
# ============================================================
section("BONUS: Heatmap — Traffic x Weather")

pivot = (
    df.groupby(["Road_traffic_density", "Weather_conditions"], observed=True)[TARGET]
    .mean()
    .unstack("Weather_conditions")
)
pivot = pivot.reindex(TRAFFIC_ORDER)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Mean Delivery Time (min)"},
            ax=ax)
ax.set_title("Mean Delivery Time: Traffic x Weather", fontsize=13, fontweight="bold")
ax.set_xlabel("Weather Condition")
ax.set_ylabel("Traffic Density")
plt.tight_layout()
save(fig, "11_heatmap_traffic_weather.png")
print("  INSIGHT: The heatmap shows that the highest mean delivery times cluster in the")
print("  'Jam' traffic row, regardless of weather type. Stormy+Jam and Fog+Jam combinations")
print("  consistently produce the longest delays, confirming that traffic is the dominant")
print("  factor and weather amplifies it rather than driving delays independently.")

# ============================================================
# Final summary table
# ============================================================
section("EDA Summary — Mean Delivery Time by Key Factors")

summary = {
    "Factor":        ["Traffic: Jam", "Traffic: High", "Traffic: Medium", "Traffic: Low",
                      "City: Semi-Urban", "City: Metropolitan", "City: Urban",
                      "Multi-delivery: 3", "Multi-delivery: 0",
                      "Vehicle: bicycle", "Vehicle: motorcycle"],
    "Mean Time(min)": [
        df[df["Road_traffic_density"] == "Jam"][TARGET].mean(),
        df[df["Road_traffic_density"] == "High"][TARGET].mean(),
        df[df["Road_traffic_density"] == "Medium"][TARGET].mean(),
        df[df["Road_traffic_density"] == "Low"][TARGET].mean(),
        df[df["City"] == "Semi-Urban"][TARGET].mean(),
        df[df["City"] == "Metropolitan"][TARGET].mean(),
        df[df["City"] == "Urban"][TARGET].mean(),
        df[df["multiple_deliveries"] == 3][TARGET].mean(),
        df[df["multiple_deliveries"] == 0][TARGET].mean(),
        df[df["Type_of_vehicle"] == "bicycle"][TARGET].mean(),
        df[df["Type_of_vehicle"] == "motorcycle"][TARGET].mean(),
    ]
}
summary_df = pd.DataFrame(summary)
summary_df["Mean Time(min)"] = summary_df["Mean Time(min)"].round(1)
print(summary_df.to_string(index=False))

print(f"\nAll plots saved to: {PLOTS_DIR}")
print("EDA complete.")
