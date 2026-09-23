"""
src/data_cleaning.py
====================
Data cleaning pipeline for the Zomato Food Delivery dataset.

Run this script directly to clean the raw CSV and save the result:
    python src/data_cleaning.py

What this script does (in order):
  1.  Load the raw CSV without modifying it.
  2.  Replace the string "NaN" with real null values.
  3.  Drop fully duplicate rows (none expected, but checked for safety).
  4.  Fix Delivery_person_Age  — cast to numeric, set age < 18 to NaN.
  5.  Fix Delivery_person_Ratings — cast to float, cap values > 5 at NaN.
  6.  Fix GPS coordinates     — set (0, 0) restaurant coords to NaN;
                                 fix negatively signed Indian lat/lon values.
  7.  Fix Order_Date           — parse to a proper datetime object.
  8.  Fix Time_Orderd          — convert Excel decimal serials to HH:MM strings;
                                 leave already-valid HH:MM values unchanged.
  9.  Fix Time_Order_picked    — same conversion as Time_Orderd plus handle
                                 overflow values like "24:05:00".
  10. Fix City                 — standardise "Metropolitian" -> "Metropolitan".
  11. Impute missing values     — median for numeric columns,
                                 mode for categorical columns.
  12. Cast multiple_deliveries  — to nullable Int64 after imputation.
  13. Cast Delivery_person_Age  — to nullable Int64 after imputation.
  14. Save cleaned data to     data/processed/zomato_cleaned.csv.
  15. Print a before/after cleaning report.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_PATH     = os.path.join("data", "raw", "Zomato Dataset.csv")
OUTPUT_PATH  = os.path.join("data", "processed", "zomato_cleaned.csv")

# ---------------------------------------------------------------------------
# Helper: convert an Excel time serial (e.g. 0.458333) to "HH:MM"
# ---------------------------------------------------------------------------
def excel_serial_to_hhmm(value: str) -> str:
    """
    Excel stores times as a fraction of 24 hours.
    Example: 0.458333... = 11:00 AM  (0.458333 * 24 = 11.0)
    We multiply by 24 to get total hours, then split into HH and MM.
    """
    try:
        decimal = float(value)
        total_minutes = round(decimal * 24 * 60)   # convert fraction → total minutes
        hours   = (total_minutes // 60) % 24        # keep hours within 0-23
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------
# Helper: fix a single time string — handles HH:MM, decimal serials, 24:xx
# ---------------------------------------------------------------------------
def clean_time_value(value) -> str:
    """
    Accepts a raw time value and returns a cleaned "HH:MM" string or NaN.

    Cases handled:
      - Already valid "HH:MM"           → returned as-is
      - Excel decimal serial "0.875"    → converted via excel_serial_to_hhmm()
      - Overflow "24:05:00"             → rolled to "00:05"
      - Literal "1" (corrupt entry)     → returned as NaN
      - Actual NaN / empty              → returned as NaN
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    # Already a valid HH:MM or HH:MM:SS
    if ":" in s:
        parts = s.split(":")
        try:
            hour   = int(parts[0])
            minute = int(parts[1])
            # Handle the "24:xx" overflow — roll over to next-day 00:xx
            if hour == 24:
                hour = 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
        except (ValueError, IndexError):
            pass
        return np.nan

    # Decimal fraction — Excel serial
    try:
        num = float(s)
        # Guard: "1" (corrupt) or any value >= 1 that is not a proper serial
        if num >= 1:
            return np.nan
        return excel_serial_to_hhmm(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
def clean_dataset(raw_path: str = RAW_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Loads the raw Zomato CSV, applies all cleaning steps, saves the result,
    and returns the cleaned DataFrame.
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Load the raw CSV
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading raw CSV ...")
    print("=" * 60)

    df = pd.read_csv(raw_path)
    original_shape = df.shape
    print(f"  Loaded {original_shape[0]:,} rows × {original_shape[1]} columns.")

    # -----------------------------------------------------------------------
    # STEP 2 — Replace the string "NaN" with real null values (np.nan)
    #
    # Why: The CSV stores missing values as the literal text "NaN" (a string).
    # Pandas will NOT treat these as null unless we replace them first.
    # This is the first thing to do before any other operation.
    # -----------------------------------------------------------------------
    print("\nSTEP 2: Replacing string 'NaN' with real null values ...")
    df = df.replace("NaN", np.nan)

    missing_before = df.isna().sum()
    print("  Missing values per column after replacement:")
    for col, n in missing_before.items():
        if n > 0:
            print(f"    {col}: {n:,}")

    # -----------------------------------------------------------------------
    # STEP 3 — Drop fully duplicate rows
    #
    # Why: Duplicate records would inflate counts and bias models.
    # From our EDA we found 0 duplicates, but we check as good practice.
    # -----------------------------------------------------------------------
    print("\nSTEP 3: Dropping duplicate rows ...")
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    print(f"  Dropped {dropped} duplicate row(s). Remaining: {len(df):,}")

    # -----------------------------------------------------------------------
    # STEP 4 — Fix Delivery_person_Age
    #
    # Why: The column was loaded as an object (string) because of "NaN" values.
    #      We cast to float first (handles NaN safely), then to Int64 (nullable int).
    #      Riders below 18 are considered invalid — likely data entry errors.
    #      We set those to NaN rather than deleting the entire row.
    # -----------------------------------------------------------------------
    print("\nSTEP 4: Fixing Delivery_person_Age ...")
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
    invalid_age = (df["Delivery_person_Age"] < 18).sum()
    df.loc[df["Delivery_person_Age"] < 18, "Delivery_person_Age"] = np.nan
    print(f"  Cast to numeric. Set {invalid_age} rows with age < 18 to NaN.")

    # -----------------------------------------------------------------------
    # STEP 5 — Fix Delivery_person_Ratings
    #
    # Why: Loaded as object due to "NaN". Cast to float.
    #      Rating scale is 1–5. Any value > 5 is invalid (we found 53 with 6.0).
    #      We set those to NaN so they can be imputed with the column median later.
    # -----------------------------------------------------------------------
    print("\nSTEP 5: Fixing Delivery_person_Ratings ...")
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
    invalid_ratings = (df["Delivery_person_Ratings"] > 5).sum()
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan
    print(f"  Cast to numeric. Set {invalid_ratings} rows with rating > 5 to NaN.")

    # -----------------------------------------------------------------------
    # STEP 6 — Fix GPS Coordinates
    #
    # Why:
    #   a) Restaurant (0.0, 0.0) — null-island coordinates. These are clearly
    #      missing/placeholder values. We set both lat and lon to NaN.
    #
    #   b) Negative Restaurant_latitude — India lies between 8°N and 37°N,
    #      so all latitudes must be positive. Negative values are sign-flipped
    #      by error. We take the absolute value.
    #
    #   c) Negative Restaurant_longitude — India's longitudes are 68°E to 97°E,
    #      all positive. Same fix: take absolute value.
    #
    #   d) Delivery_location coordinates — near-zero values (0.01–0.13) exist
    #      only for rows where restaurant coords were (0,0). These are invalid
    #      placeholders too. Set to NaN alongside the restaurant coords.
    # -----------------------------------------------------------------------
    print("\nSTEP 6: Fixing GPS coordinates ...")

    # Convert coordinate columns to numeric first
    for col in ["Restaurant_latitude", "Restaurant_longitude",
                "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # a) Restaurant (0, 0) → NaN for both restaurant and delivery coords
    zero_mask = (df["Restaurant_latitude"] == 0) & (df["Restaurant_longitude"] == 0)
    count_zero = zero_mask.sum()
    df.loc[zero_mask, ["Restaurant_latitude", "Restaurant_longitude",
                        "Delivery_location_latitude", "Delivery_location_longitude"]] = np.nan
    print(f"  Set {count_zero:,} rows with restaurant (0,0) coords to NaN.")

    # b & c) Negative lat/lon → flip sign to positive
    neg_lat = (df["Restaurant_latitude"] < 0).sum()
    neg_lon = (df["Restaurant_longitude"] < 0).sum()
    df["Restaurant_latitude"]  = df["Restaurant_latitude"].abs()
    df["Restaurant_longitude"] = df["Restaurant_longitude"].abs()
    print(f"  Fixed {neg_lat} negative Restaurant_latitude values (absolute value applied).")
    print(f"  Fixed {neg_lon} negative Restaurant_longitude values (absolute value applied).")

    # -----------------------------------------------------------------------
    # STEP 7 — Fix Order_Date
    #
    # Why: The date is stored as a string "DD-MM-YYYY". We parse it into a
    #      proper Python datetime so we can later extract day-of-week, month, etc.
    # -----------------------------------------------------------------------
    print("\nSTEP 7: Parsing Order_Date to datetime ...")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    unparseable = df["Order_Date"].isna().sum()
    print(f"  Parsed. {unparseable} unparseable dates set to NaT.")

    # -----------------------------------------------------------------------
    # STEP 8 — Fix Time_Orderd (order placement time)
    #
    # Why: ~3,638 rows contain Excel decimal serials (e.g. 0.875) instead of
    #      HH:MM strings because the original spreadsheet stored times as
    #      fractions of a day. We convert those to "HH:MM" strings.
    #      Rows that are NaN stay NaN (1,731 rows).
    # -----------------------------------------------------------------------
    print("\nSTEP 8: Fixing Time_Orderd ...")
    df["Time_Orderd"] = df["Time_Orderd"].apply(clean_time_value)
    remaining_bad = df["Time_Orderd"].isna().sum()
    print(f"  Cleaned. {remaining_bad:,} values remain NaN (were already missing).")

    # -----------------------------------------------------------------------
    # STEP 9 — Fix Time_Order_picked (pickup time)
    #
    # Why: Same Excel serial issue (~5,007 rows). Also contains "24:05:00"
    #      overflow values (hour 24 does not exist — rolled to 00:xx)
    #      and the literal value "1" which is corrupt.
    # -----------------------------------------------------------------------
    print("\nSTEP 9: Fixing Time_Order_picked ...")
    df["Time_Order_picked"] = df["Time_Order_picked"].apply(clean_time_value)
    remaining_bad_pick = df["Time_Order_picked"].isna().sum()
    print(f"  Cleaned. {remaining_bad_pick:,} values remain NaN after conversion.")

    # -----------------------------------------------------------------------
    # STEP 10 — Standardise City column
    #
    # Why: "Metropolitian" is a consistent misspelling of "Metropolitan"
    #      throughout the dataset (affects 34,087 rows). Correcting it ensures
    #      consistent grouping in analysis and encoding.
    # -----------------------------------------------------------------------
    print("\nSTEP 10: Standardising City values ...")
    df["City"] = df["City"].str.strip()
    df["City"] = df["City"].replace("Metropolitian", "Metropolitan")
    print("  Replaced 'Metropolitian' -> 'Metropolitan'.")

    # -----------------------------------------------------------------------
    # STEP 11 — Impute missing values
    #
    # Strategy:
    #   Numeric columns  → fill with the column MEDIAN
    #     (median is preferred over mean because it is robust to outliers)
    #   Categorical cols → fill with the column MODE (most frequent value)
    #
    # Columns affected:
    #   Numeric:      Delivery_person_Age, Delivery_person_Ratings,
    #                 multiple_deliveries,
    #                 Restaurant_latitude, Restaurant_longitude,
    #                 Delivery_location_latitude, Delivery_location_longitude
    #   Categorical:  Weather_conditions, Road_traffic_density, Festival, City
    #
    # Note: Time_Orderd and Time_Order_picked are kept as NaN where missing.
    #       They will be used to engineer a "pickup wait time" feature later;
    #       rows with both times missing simply won't have that feature.
    # -----------------------------------------------------------------------
    print("\nSTEP 11: Imputing missing values ...")

    numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "multiple_deliveries",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]

    categorical_cols = [
        "Weather_conditions",
        "Road_traffic_density",
        "Festival",
        "City",
    ]

    for col in numeric_cols:
        median_val = df[col].median()
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: filled {n_filled:,} NaN with median={median_val:.4f}")

    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        n_filled = df[col].isna().sum()
        df[col] = df[col].fillna(mode_val)
        print(f"  {col}: filled {n_filled:,} NaN with mode='{mode_val}'")

    # -----------------------------------------------------------------------
    # STEP 12 — Cast multiple_deliveries to integer
    #
    # Why: After imputation it is a float (median may be 1.0 etc.).
    #      Conceptually it is a count (0, 1, 2, 3) so int is more appropriate.
    # -----------------------------------------------------------------------
    print("\nSTEP 12: Casting multiple_deliveries to integer ...")
    # Use pandas nullable Int64 (capital I) which safely handles any remaining NaN.
    df["multiple_deliveries"] = df["multiple_deliveries"].round().astype("Int64")
    print("  Done.")

    # -----------------------------------------------------------------------
    # STEP 13 — Final type enforcement
    #
    # Cast Age to nullable Int64 now that all NaN values have been filled.
    # -----------------------------------------------------------------------
    df["Delivery_person_Age"] = df["Delivery_person_Age"].round().astype("Int64")

    # -----------------------------------------------------------------------
    # STEP 14 — Save cleaned dataset
    # -----------------------------------------------------------------------
    print(f"\nSTEP 14: Saving cleaned dataset to '{output_path}' ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows × {len(df.columns)} columns.")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Original shape : {original_shape[0]:,} rows × {original_shape[1]} columns")
    print(f"  Cleaned shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Rows removed   : {original_shape[0] - df.shape[0]}")
    print(f"  Remaining NaN  : {df.isna().sum().sum()}")
    print(f"  Output file    : {output_path}")
    print("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clean_dataset()
