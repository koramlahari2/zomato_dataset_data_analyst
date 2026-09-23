"""
src/features.py
===============
Feature engineering pipeline for the Zomato Food Delivery dataset.

Run from the project root:
    python src/features.py

Input  : data/processed/zomato_cleaned.csv
Output : data/processed/zomato_features.csv

Features created
----------------
From GPS coordinates
  distance_km           — Haversine straight-line distance between restaurant
                          and delivery address (km).

From Order_Date
  order_day_of_week     — 0=Monday … 6=Sunday (captures weekly demand cycles).
  order_month           — 2=Feb, 3=Mar, 4=Apr (captures monthly seasonality).
  is_weekend            — 1 if Saturday or Sunday, 0 otherwise.

From Time_Orderd (order placement time)
  order_hour            — Hour of day the customer placed the order (0–23).
  is_peak_hour          — 1 if order was placed during lunch (11-14) or
                          dinner rush (18-22), 0 otherwise.

From Time_Order_picked (restaurant pickup time)
  pickup_wait_min       — Minutes between order placement and restaurant pickup.
                          Captures kitchen prep + rider response time. Only
                          computed when both time columns are valid; otherwise NaN.

Encoding categorical variables
  traffic_encoded       — Ordinal: Low=1, Medium=2, High=3, Jam=4.
                          Preserves the natural order of traffic severity.
  weather_encoded       — Ordinal severity: Sunny=1, Cloudy=2, Windy=3,
                          Fog=4, Sandstorms=5, Stormy=6.
  city_encoded          — Ordinal: Urban=1, Metropolitan=2, Semi-Urban=3.
                          Reflects typical route complexity / distance.
  vehicle_encoded       — Nominal label-encoded:
                          electric_scooter=0, motorcycle=1, scooter=2, bicycle=3.
  order_type_encoded    — Nominal label-encoded:
                          Buffet=0, Drinks=1, Meal=2, Snack=3.
  festival_encoded      — Binary: No=0, Yes=1.

Columns dropped before saving
  ID, Delivery_person_ID      — identifiers, no predictive value for a general model
  Order_Date                  — replaced by day_of_week, month, is_weekend
  Time_Orderd                 — replaced by order_hour, is_peak_hour
  Time_Order_picked           — replaced by pickup_wait_min
  Raw lat/lon columns         — replaced by distance_km
  Raw categorical strings     — replaced by encoded numeric columns

No data leakage
  pickup_wait_min uses only order placement + pickup times, both of which are
  known before the delivery completes.  Time_taken (min) is never used to
  construct any feature.
"""

import os
import math
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEANED_CSV  = os.path.join("data", "processed", "zomato_cleaned.csv")
OUTPUT_PATH  = os.path.join("data", "processed", "zomato_features.csv")
TARGET       = "Time_taken (min)"

# ---------------------------------------------------------------------------
# Helper: Haversine distance in km
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance between two GPS points in kilometres.
    Returns NaN if any coordinate is NaN.
    """
    if any(math.isnan(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Helper: parse "HH:MM" to total minutes since midnight
# ---------------------------------------------------------------------------
def hhmm_to_minutes(value) -> float:
    """
    Converts a "HH:MM" string to total minutes since midnight.
    Returns NaN for missing or unparseable values.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) < 2:
        return np.nan
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Helper: compute pickup wait time in minutes, handling overnight orders
# ---------------------------------------------------------------------------
def pickup_wait(ordered_min: float, picked_min: float) -> float:
    """
    Returns (picked - ordered) in minutes.
    Handles the overnight case: if the difference is negative,
    the order was placed before midnight and picked up after midnight —
    we add 1440 (24*60) to correct for the day rollover.
    Returns NaN if either input is NaN.
    """
    if pd.isna(ordered_min) or pd.isna(picked_min):
        return np.nan
    diff = picked_min - ordered_min
    if diff < 0:
        diff += 1440   # overnight order
    # Sanity cap: if wait > 180 min it is almost certainly a data error
    if diff > 180:
        return np.nan
    return diff


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------
def build_features(cleaned_path: str = CLEANED_CSV,
                   output_path: str  = OUTPUT_PATH) -> pd.DataFrame:

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading cleaned dataset ...")
    print("=" * 60)
    df = pd.read_csv(cleaned_path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # -----------------------------------------------------------------------
    # FEATURE 1 — Haversine delivery distance (km)
    #
    # Why it helps: longer routes take more time. Distance is a direct
    # physical driver of delivery duration independent of other factors.
    # No leakage: both restaurant and customer locations are known at order
    # placement time.
    # -----------------------------------------------------------------------
    print("\nFEATURE 1: Calculating Haversine delivery distance ...")
    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r["Restaurant_latitude"],        r["Restaurant_longitude"],
            r["Delivery_location_latitude"], r["Delivery_location_longitude"]
        ),
        axis=1
    )
    n_nan = df["distance_km"].isna().sum()
    print(f"  distance_km — mean={df['distance_km'].mean():.2f} km  "
          f"range=[{df['distance_km'].min():.2f}, {df['distance_km'].max():.2f}]  "
          f"NaN={n_nan}")

    # -----------------------------------------------------------------------
    # FEATURE 2 — Order day of week (0=Monday, 6=Sunday)
    #
    # Why it helps: weekdays and weekends have different demand patterns.
    # Fridays and Sundays often have higher order volumes which may stress
    # the delivery network and increase wait times.
    # -----------------------------------------------------------------------
    print("\nFEATURE 2: Extracting order_day_of_week ...")
    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek
    print(f"  Distribution:\n{df['order_day_of_week'].value_counts().sort_index().to_string()}")

    # -----------------------------------------------------------------------
    # FEATURE 3 — Order month
    #
    # Why it helps: the dataset spans Feb–Apr 2022. Month captures any
    # seasonal or festival-period effects that vary across the three months.
    # -----------------------------------------------------------------------
    print("\nFEATURE 3: Extracting order_month ...")
    df["order_month"] = df["Order_Date"].dt.month
    print(f"  Unique months: {sorted(df['order_month'].dropna().unique().tolist())}")

    # -----------------------------------------------------------------------
    # FEATURE 4 — Weekend indicator
    #
    # Why it helps: weekend demand spikes are a well-known pattern in food
    # delivery. A binary flag is easier for tree-based models to use than
    # the raw day-of-week integer.
    # -----------------------------------------------------------------------
    print("\nFEATURE 4: Creating is_weekend flag ...")
    df["is_weekend"] = (df["order_day_of_week"] >= 5).astype(int)
    print(f"  Weekday orders: {(df['is_weekend']==0).sum():,}  |  "
          f"Weekend orders: {(df['is_weekend']==1).sum():,}")

    # -----------------------------------------------------------------------
    # FEATURE 5 — Order hour (0–23)
    #
    # Why it helps: hour of day strongly influences traffic and kitchen load.
    # Lunch (12–14) and dinner (19–22) are peak periods with higher delays.
    # -----------------------------------------------------------------------
    print("\nFEATURE 5: Extracting order_hour from Time_Orderd ...")
    df["_ordered_min"] = df["Time_Orderd"].apply(hhmm_to_minutes)
    df["order_hour"] = df["_ordered_min"].apply(
        lambda m: int(m // 60) if not pd.isna(m) else np.nan
    )
    n_nan_hour = df["order_hour"].isna().sum()
    valid = df["order_hour"].dropna()
    print(f"  order_hour — range=[{int(valid.min())}, {int(valid.max())}]  NaN={n_nan_hour:,}")

    # Fill the NaN hours with the median hour so the feature is fully populated
    median_hour = valid.median()
    df["order_hour"] = df["order_hour"].fillna(median_hour).round().astype(int)
    print(f"  Filled {n_nan_hour:,} NaN with median hour={int(median_hour)}")

    # -----------------------------------------------------------------------
    # FEATURE 6 — Peak hour indicator
    #
    # Why it helps: a binary flag cleanly captures whether the order falls
    # inside a known rush window. Simpler for linear models than the raw hour.
    # Lunch rush: 11:00–14:59  |  Dinner rush: 18:00–22:59
    # -----------------------------------------------------------------------
    print("\nFEATURE 6: Creating is_peak_hour flag ...")
    df["is_peak_hour"] = (
        df["order_hour"].between(11, 14) | df["order_hour"].between(18, 22)
    ).astype(int)
    print(f"  Peak-hour orders : {df['is_peak_hour'].sum():,}  "
          f"({df['is_peak_hour'].mean()*100:.1f}%)")

    # -----------------------------------------------------------------------
    # FEATURE 7 — Pickup wait time (minutes)
    #
    # Why it helps: the gap between when the customer ordered and when the
    # rider picked up from the restaurant reflects kitchen prep speed and
    # rider proximity. A longer wait here often extends total delivery time.
    # No leakage: both timestamps precede the delivery completion.
    # -----------------------------------------------------------------------
    print("\nFEATURE 7: Computing pickup_wait_min ...")
    df["_picked_min"] = df["Time_Order_picked"].apply(hhmm_to_minutes)
    df["pickup_wait_min"] = df.apply(
        lambda r: pickup_wait(r["_ordered_min"], r["_picked_min"]), axis=1
    )
    n_nan_wait = df["pickup_wait_min"].isna().sum()
    valid_wait = df["pickup_wait_min"].dropna()
    print(f"  pickup_wait_min — mean={valid_wait.mean():.1f} min  "
          f"range=[{valid_wait.min():.0f}, {valid_wait.max():.0f}]  "
          f"NaN={n_nan_wait:,}")

    # Fill remaining NaN with the median pickup wait
    median_wait = valid_wait.median()
    df["pickup_wait_min"] = df["pickup_wait_min"].fillna(median_wait).round(1)
    print(f"  Filled {n_nan_wait:,} NaN with median wait={median_wait:.1f} min")

    # -----------------------------------------------------------------------
    # FEATURE 8 — Traffic density ordinal encoding
    #
    # Why ordinal: traffic has a natural severity order (Low < Medium < High
    # < Jam). Encoding this order as integers 1–4 preserves that information.
    # -----------------------------------------------------------------------
    print("\nFEATURE 8: Encoding Road_traffic_density (ordinal) ...")
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    df["traffic_encoded"] = df["Road_traffic_density"].map(traffic_map)
    print(f"  Mapping: {traffic_map}")

    # -----------------------------------------------------------------------
    # FEATURE 9 — Weather condition ordinal encoding
    #
    # Why ordinal: weather conditions can be ordered by how much they
    # are expected to impede travel (Sunny = least impact, Stormy = most).
    # This encoding was informed by the EDA mean-time rankings.
    # -----------------------------------------------------------------------
    print("\nFEATURE 9: Encoding Weather_conditions (ordinal severity) ...")
    weather_map = {
        "Sunny": 1, "Cloudy": 2, "Windy": 3,
        "Fog": 4, "Sandstorms": 5, "Stormy": 6
    }
    df["weather_encoded"] = df["Weather_conditions"].map(weather_map)
    print(f"  Mapping: {weather_map}")

    # -----------------------------------------------------------------------
    # FEATURE 10 — City tier ordinal encoding
    #
    # Why ordinal: city tier reflects route complexity and typical delivery
    # distance. Urban = densest / shortest, Metropolitan = medium,
    # Semi-Urban = largest distances. EDA confirmed this ordering.
    # -----------------------------------------------------------------------
    print("\nFEATURE 10: Encoding City (ordinal by route complexity) ...")
    city_map = {"Urban": 1, "Metropolitan": 2, "Semi-Urban": 3}
    df["city_encoded"] = df["City"].map(city_map)
    print(f"  Mapping: {city_map}")

    # -----------------------------------------------------------------------
    # FEATURE 11 — Vehicle type label encoding (nominal)
    #
    # Why nominal (not ordinal): there is no natural speed order among vehicle
    # types that is consistent across all contexts, so we use simple label
    # encoding. Tree-based models handle this well.
    # -----------------------------------------------------------------------
    print("\nFEATURE 11: Encoding Type_of_vehicle (label encoding) ...")
    vehicle_map = {"electric_scooter": 0, "motorcycle": 1, "scooter": 2, "bicycle": 3}
    df["vehicle_encoded"] = df["Type_of_vehicle"].map(vehicle_map)
    print(f"  Mapping: {vehicle_map}")

    # -----------------------------------------------------------------------
    # FEATURE 12 — Order type label encoding (nominal)
    #
    # Why nominal: EDA showed no meaningful time ordering across order types;
    # they are categories, not a scale.
    # -----------------------------------------------------------------------
    print("\nFEATURE 12: Encoding Type_of_order (label encoding) ...")
    order_map = {"Buffet": 0, "Drinks": 1, "Meal": 2, "Snack": 3}
    df["order_type_encoded"] = df["Type_of_order"].map(order_map)
    print(f"  Mapping: {order_map}")

    # -----------------------------------------------------------------------
    # FEATURE 13 — Festival binary encoding
    #
    # Why it helps: festival periods increase demand, strain the delivery
    # network, and often cause traffic spikes — all of which raise delivery
    # times.
    # -----------------------------------------------------------------------
    print("\nFEATURE 13: Encoding Festival (binary) ...")
    festival_map = {"No": 0, "Yes": 1}
    df["festival_encoded"] = df["Festival"].map(festival_map)
    print(f"  Festival orders: {df['festival_encoded'].sum():,}  "
          f"({df['festival_encoded'].mean()*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Drop columns that have been replaced or are identifiers
    # -----------------------------------------------------------------------
    print("\nDropping raw/identifier columns ...")
    cols_to_drop = [
        # Identifiers
        "ID", "Delivery_person_ID",
        # Replaced by engineered date/time features
        "Order_Date", "Time_Orderd", "Time_Order_picked",
        # Internal helpers
        "_ordered_min", "_picked_min",
        # Replaced by distance_km
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        # Replaced by encoded columns
        "Weather_conditions", "Road_traffic_density",
        "City", "Type_of_vehicle", "Type_of_order", "Festival",
    ]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns. Remaining: {df.shape[1]}")

    # -----------------------------------------------------------------------
    # Final column order — features first, target last
    # -----------------------------------------------------------------------
    feature_cols = [
        "distance_km",
        "order_hour", "order_day_of_week", "order_month", "is_weekend", "is_peak_hour",
        "pickup_wait_min",
        "Delivery_person_Age", "Delivery_person_Ratings",
        "multiple_deliveries",
        "Vehicle_condition",
        "traffic_encoded", "weather_encoded", "city_encoded",
        "vehicle_encoded", "order_type_encoded", "festival_encoded",
    ]
    df = df[feature_cols + [TARGET]]

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    print("\nFinal NaN check:")
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if nan_cols.empty:
        print("  No missing values — dataset is fully populated.")
    else:
        print(nan_cols.to_string())

    print(f"\nFinal shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nColumn list:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        print(f"  {i:2d}. {col:<30s} dtype={dtype}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved feature dataset to: {output_path}")

    return df


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_features()
