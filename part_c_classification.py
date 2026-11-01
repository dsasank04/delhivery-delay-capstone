"""Part C - delay classifier, feature-signal check, tuning and the alert threshold.
Run from the repository root after Part A:  python part_c_classification.py"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# 1. Stratified 70 / 15 / 15 split.
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes:", len(X_train), len(X_val), len(X_test), "| delayed:", y_train.sum(), y_val.sum(), y_test.sum())


def scores_from_cells(y_true, y_pred):
    """Confusion-matrix cells plus the four scores, all derived from the same cells."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": (tn + tp) / (tn + fp + fn + tp), "precision": precision, "recall": recall, "f1": f1}


# 2. Blind baseline and three classifiers at the default 0.5 rule.
model_rows = [{"model": "blind_all_on_time", **scores_from_cells(y_val, pd.Series(0, index=y_val.index))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **scores_from_cells(y_val, logreg.predict(X_val))})

best_tree = None
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    row = scores_from_cells(y_val, tree.predict(X_val))
    if best_tree is None or row["f1"] > best_tree[1]["f1"]:  # strict > keeps the smaller depth on ties
        best_tree = (d, row)
model_rows.append({"model": f"decision_tree_depth_{best_tree[0]}", **best_tree[1]})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **scores_from_cells(y_val, forest.predict(X_val))})
models = pd.DataFrame(model_rows).round(4)
models.to_csv("outputs/part_c_models.csv", index=False)

# 3. Which columns carry signal?
f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
feature_scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                               "rf_importance": forest.feature_importances_})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="first").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": scores_from_cells(y_val, pipe.predict(X_val))["f1"]})
pd.DataFrame(selectk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. Tuning on the 7 selected columns (selector fitted on training rows only; test rows untouched).
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
X_train_k = selector.transform(X_train)
X_val_k = selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(X_train_k, y_train)
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5,
                                   random_state=42).fit(X_train_k, y_train)
default_7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_k, y_train)


def ordered_params(params):
    return {key: params[key] for key in ["n_estimators", "max_depth", "min_samples_leaf"]}


tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": len(grid_search.cv_results_["params"]), "cv_folds": 5,
     "best_params": ordered_params(grid_search.best_params_), "best_cv_accuracy": grid_search.best_score_,
     "val_f1": scores_from_cells(y_val, grid_search.predict(X_val_k))["f1"]},
    {"search": "randomized_search", "candidates": len(random_search.cv_results_["params"]), "cv_folds": 5,
     "best_params": ordered_params(random_search.best_params_), "best_cv_accuracy": random_search.best_score_,
     "val_f1": scores_from_cells(y_val, random_search.predict(X_val_k))["f1"]},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
     "best_params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}, "best_cv_accuracy": None,
     "val_f1": scores_from_cells(y_val, default_7.predict(X_val_k))["f1"]},
])
tuning["total_fits"] = [c * f if f else 1 for c, f in zip(tuning["candidates"], tuning["cv_folds"])]
tuning = tuning[["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]]
tuning.round(4).to_csv("outputs/part_c_tuning.csv", index=False)

# 5. Final configuration: highest val_f1; ties -> default, then grid, then randomized.
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
by_priority = tuning.set_index("search").loc[priority]
chosen_from = by_priority["val_f1"].round(4).idxmax()  # idxmax returns the first maximum in priority order
chosen = by_priority.loc[chosen_from, "best_params"]
final_model = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                        ("model", RandomForestClassifier(random_state=42, **chosen))]).fit(X_train, y_train)

# 6. Alert threshold from validation probabilities.
val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = [{"threshold": t, **scores_from_cells(y_val, (val_proba >= t).astype(int))}
                  for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
thresholds = pd.DataFrame(threshold_rows)
thresholds.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
meets_recall = thresholds.loc[thresholds["recall"] >= 0.80, "threshold"]
alert_threshold = float(meets_recall.max()) if len(meets_recall) else 0.1

# 7. One test evaluation at the chosen threshold.
test_proba = final_model.predict_proba(X_test)[:, 1]
final_test = pd.DataFrame([{"threshold": alert_threshold,
                            **scores_from_cells(y_test, (test_proba >= alert_threshold).astype(int))}])
final_test.round(4).to_csv("outputs/part_c_final_test.csv", index=False)

final_config = pd.DataFrame([{"selector_k": K, "n_estimators": chosen["n_estimators"],
                              "max_depth": "unlimited" if chosen["max_depth"] is None else chosen["max_depth"],
                              "min_samples_leaf": chosen["min_samples_leaf"], "threshold": alert_threshold,
                              "chosen_from": chosen_from}])
final_config.to_csv("outputs/part_c_final_config.csv", index=False)

print(models.to_string(index=False))
print(feature_scores.round(4).to_string(index=False))
print("noise share of forest importance:",
      round(feature_scores.loc[feature_scores["is_noise"], "rf_importance"].sum(), 4))
print(pd.DataFrame(selectk_rows).round(4).to_string(index=False))
print(tuning.round(4).to_string(index=False))
print(thresholds.round(4).to_string(index=False))
print(final_test.round(4).to_string(index=False))
print(final_config.to_string(index=False))
