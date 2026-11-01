"""Part A - clean the raw export, build the modelling table and the first delay evidence.
Run from the repository root:  python part_a_clean_eda.py"""
import os

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "Before" counts: measured after removing duplicates and before any filling.
dedup = raw.drop_duplicates()
weight_before = pd.to_numeric(dedup["parcel_weight_kg"], errors="coerce")
spellings_before = dedup["service_type"].nunique()
spellings_after = dedup["service_type"].str.strip().str.title().nunique()

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
table.to_csv("data/modelling_table.csv", index=False)

# Delivery-hours outliers by the 1.5 x IQR rule (bounds from the unrounded quartiles).
q1, q3 = table["delivery_hours"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_outliers = int(((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum())

check_rows = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", int(dedup["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(weight_before.isna().sum())),
    ("sentinel_weight_rows", int((weight_before > 100).sum())),
    ("missing_declared_value_before", int(dedup["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup["payment_mode"].isna().sum())),
    ("service_type_spellings_before", spellings_before),
    ("service_type_spellings_after", spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(table["is_delayed"].sum())),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", n_outliers),
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", table["log_declared_value"].mean()),
    ("log_declared_value_median", table["log_declared_value"].median()),
]
checks = pd.DataFrame({"metric": [m for m, _ in check_rows],
                       "value": pd.Series([v if isinstance(v, (int, np.integer)) else round(float(v), 4)
                                           for _, v in check_rows], dtype=object)})
checks.to_csv("outputs/part_a_checks.csv", index=False)

# Delay evidence by service type, hub tier and distance band.
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"], right=True)
groups = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = (table.groupby(dim, observed=True)
         .agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
         .reset_index().rename(columns={dim: "group"}))
    g.insert(0, "dimension", dim)
    groups.append(g)
by_group = pd.concat(groups, ignore_index=True)
by_group["group"] = by_group["group"].astype(str)
by_group.round(4).to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Correlation matrix of the numeric columns of the modelling table.
corr = table.drop(columns=["distance_band"]).select_dtypes(include=np.number).corr()
corr.round(4).rename_axis("column").reset_index().to_csv("outputs/part_a_correlations.csv", index=False)

print(checks.to_string(index=False))
print(by_group.round(4).to_string(index=False))
print("promised hours used:", PROMISED_HOURS)
