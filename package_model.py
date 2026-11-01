"""Package Part C's final classifier as a reusable artefact in models/.
Run from the repository root after Parts A and C:  python package_model.py"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, build_features, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
threshold = float(config["threshold"])

# Rebuild Part C's stratified split with the same settings.
table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

pipeline = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42, n_estimators=int(config["n_estimators"]),
                                     max_depth=max_depth, min_samples_leaf=int(config["min_samples_leaf"]))),
]).fit(X_train, y_train)

# The medians are the fill-in values clean_shipments computes on the full historical table.
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))

joblib.dump(pipeline, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}).to_csv(
    "models/impute_values.csv", index=False)

# Reload check: the pickle must reproduce the in-memory probabilities.
reloaded = joblib.load("models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - pipeline.predict_proba(X_test)[:, 1]).max()
reload_check = pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
                             "value": pd.Series([len(X_test), round(float(diff), 4), threshold], dtype=object)})
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

# Sanity check that the stored medians clean the new parcels without gaps.
new_clean, _ = clean_shipments(pd.read_csv("data/new_shipments.csv"), medians=medians)
new_features = build_features(new_clean, pd.read_csv("data/hubs.csv"))
assert new_features[FEATURES].isna().sum().sum() == 0

print(reload_check.to_string(index=False))
