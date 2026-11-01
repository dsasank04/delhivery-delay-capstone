"""Shared constants and cleaning / feature functions (used by Part A and predict_delay.py)."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_ABOVE = 100  # the network accepts parcels up to 40 kg; larger readings are scanner faults


def clean_shipments(raw, medians=None):
    """Return (clean_df, medians)."""
    df = raw.copy()

    # Step 1 - drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Step 2 - service_type: strip spaces, title case (3 spellings should remain)
    df["service_type"] = df["service_type"].str.strip().str.title()

    # Step 3 - payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step 4 - declared_value: remove the rupee sign and commas, to number,
    #          store in declared_value_inr, drop the text column
    text = df["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(text, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])

    # Step 5 - dispatch_time: to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")

    # Step 6 - parcel_weight_kg above 100 -> missing (scanner sentinel)
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_ABOVE, "parcel_weight_kg"] = np.nan

    # Step 7 - fill distance_km, parcel_weight_kg, declared_value_inr with medians
    #          (compute them here only if medians is None; values must be float)
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].astype(float).fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and model features. Must not use delivery_hours."""
    df = clean.copy()

    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = df.merge(hubs[hub_cols], on="hub_id", how="left", validate="many_to_one")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
