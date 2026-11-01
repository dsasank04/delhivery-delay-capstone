"""Part B - delivery-hours regression.
Run from the repository root after Part A:  python part_b_regression.py"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# 1. Split 70 / 15 / 15.
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. Scaler fitted on the training rows only.
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
check_rows = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", float(np.abs(train_scaled.mean(axis=0)).max())),
    ("test_scaled_max_abs_mean", float(np.abs(test_scaled.mean(axis=0)).max())),
]
checks = pd.DataFrame({"metric": [m for m, _ in check_rows],
                       "value": pd.Series([v if isinstance(v, int) else round(v, 4) for _, v in check_rows], dtype=object)})
checks.to_csv("outputs/part_b_checks.csv", index=False)


def regression_row(name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# 3. Baseline: the training mean for every validation row.
rows = [regression_row("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

# 4. Tree-depth diagnostic.
depth_rows = []
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    depth_rows.append({"max_depth": "unlimited" if d is None else d,
                       "train_rmse": rmse(y_train, tree.predict(X_train)),
                       "val_rmse": rmse(y_val, tree.predict(X_val)), "_depth": d})
depth_table = pd.DataFrame(depth_rows)
best_depth = depth_table.loc[depth_table["val_rmse"].idxmin(), "_depth"]  # idxmin keeps the first (smaller) depth on ties
best_depth = None if pd.isna(best_depth) else int(best_depth)
depth_table.drop(columns="_depth").round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. Four candidate models, scored on validation.
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    row = regression_row(name, "val", y_val, model.predict(X_val))
    val_rmse[name] = row["RMSE"]
    rows.append(row)

# 6. Learning-rate experiment.
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. Cross-validate the two best candidates on train + validation with one shared KFold.
ranking = sorted(val_rmse, key=val_rmse.get)
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranking[:2]:
    scores = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

# Final model = lowest validation RMSE, already fitted on the training rows; scored once on test.
final_name = ranking[0]
rows.append(regression_row(final_name, "test", y_test, candidates[final_name].predict(X_test)))
metrics = pd.DataFrame(rows).round(4)
metrics.to_csv("outputs/part_b_metrics.csv", index=False)

print(checks.to_string(index=False))
print(depth_table.drop(columns="_depth").round(4).to_string(index=False))
print(metrics.to_string(index=False))
print(pd.DataFrame(lr_rows).round(4).to_string(index=False))
print(pd.DataFrame(cv_rows).round(4).to_string(index=False))
print("validation ranking:", ranking, "| best tree depth:", best_depth)
