"""Part D - hub profiles, K-Means segments and the tier-versus-segment check.
Run from the repository root after Part A:  python part_d_segments.py"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. Hub profile, one row per hub in ascending hub_id order.
profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index().sort_values("hub_id").reset_index(drop=True))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. Scale the four profile columns and cluster into the three operating playbooks.
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
profile["cluster"] = kmeans.labels_


# 3. Separation evidence: mean across-cluster distance over mean within-cluster distance (276 hub pairs).
def across_over_within(points, labels):
    across, within = [], []
    for i, j in combinations(range(len(points)), 2):
        dist = float(np.linalg.norm(points[i] - points[j]))
        (within if labels[i] == labels[j] else across).append(dist)
    return np.mean(across) / np.mean(within)


separation_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled).labels_
    sizes = sorted(np.bincount(labels), reverse=True)
    separation_rows.append({"K": k, "across_over_within": across_over_within(scaled, labels),
                            "cluster_sizes": "-".join(str(s) for s in sizes)})
separation = pd.DataFrame(separation_rows)
separation.round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. Name clusters by their mean hub delay rate: Standard < Watch < Escalate.
cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
names = dict(zip(cluster_order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "size"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                        delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                        mean_n_stops=("mean_n_stops", "mean"))
                   .reindex(["Standard", "Watch", "Escalate"]).reset_index())
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_SEGMENT)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(profile.round(4).to_string(index=False))
print(separation.round(4).to_string(index=False))
print(segment_profile.round(4).to_string(index=False))
print(mismatch.to_string(index=False))
