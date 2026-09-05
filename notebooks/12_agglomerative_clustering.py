from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

INPUT_FILE = Path("data/processed/district_modelling_features.csv")

TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")

METRICS_FILE = TABLE_DIR / "table_agglomerative_clustering_metrics.csv"
AGGLOMERATIVE_VALIDATION_FILE = (TABLE_DIR / "table_agglomerative_output_validation.csv")
CLUSTER_OUTPUT_FILE = Path("data/processed/district_agglomerative_clusters.csv")
MODEL_SELECTION_SUMMARY_FILE = (TABLE_DIR / "table_agglomerative_model_selection_summary.csv")
CLUSTER_PROFILE_FILE = TABLE_DIR / "table_agglomerative_cluster_profile.csv"
CLUSTER_MEMBERSHIP_FILE = TABLE_DIR/ "table_agglomerative_cluster_membership.csv"
PCA_OUTPUT_FILE = TABLE_DIR / "table_agglomerative_pca_coordinates.csv"
PCA_FIGURE_FILE = FIGURE_DIR / "figure_agglomerative_clusters_pca.png"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_DISTRICT_COUNT = 32

model_df = pd.read_csv(INPUT_FILE, dtype={"district_code": str})

print("Modelling dataset shape:")
print(model_df.shape)

modelling_features = [
    "child_share_percent",
    "elderly_share_percent",
    "rural_share_percent",
    "vaccination_gap_percent",
    "wash_access_gap_score",
    "child_health_burden_score",
    "maternal_care_gap_score",
    "newborn_followup_gap_score",
    "ncd_burden_score",
    "breast_cancer_screening_gap_percent",
]


# Missing Value check
missing_values = model_df[modelling_features].isna().sum()

print("\nMissing values:")
print(missing_values.to_string())

if missing_values.sum() > 0:
    raise ValueError("Missing values found in modelling features. Fix before clustering.")

# Scale Features
X = model_df[modelling_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test K values
metrics_records = []

for k in range(2, 6):
    agg_model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")

    labels = agg_model.fit_predict(X_scaled)

    cluster_sizes = pd.Series(labels).value_counts()

    metrics_records.append({
        "k": k,
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "minimum_cluster_size": int(cluster_sizes.min()),
        "maximum_cluster_size": int(cluster_sizes.max()),
    })

metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(METRICS_FILE, index=False)

print("\nAgglomerative clustering metrics:")
print(metrics_df.to_string(index=False))

print("\nSaved Agglomerative metrics to:")
print(METRICS_FILE)

eligible_metrics_df = metrics_df.loc[metrics_df["minimum_cluster_size"] >= 2].copy()

if eligible_metrics_df.empty:
    raise ValueError(
        "No Agglomerative candidate produced clusters "
        "with at least two districts."
    )

best_metrics_row = (eligible_metrics_df.sort_values(by=["silhouette_score", "davies_bouldin_score",], ascending=[False, True],).iloc[0])

best_k = int(best_metrics_row["k"])

print("\nProvisional best k:")
print(best_k)

final_agg_model = AgglomerativeClustering(n_clusters=best_k, metric="euclidean", linkage="ward",)

model_df["agglomerative_cluster"] = (final_agg_model.fit_predict(X_scaled))

cluster_counts = (model_df.groupby("agglomerative_cluster").size().rename("district_count"))

agglomerative_validation_records = [
    {
        "check": "row_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(model_df),
    },
    {
        "check": "unique_district_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(model_df["district_name"].nunique()),
    },
    {
        "check": "observed_cluster_count",
        "expected": best_k,
        "observed": int(model_df["agglomerative_cluster"].nunique()),
    },
    {
        "check": "exact_cluster_labels",
        "expected": list(range(best_k)),
        "observed": sorted(
            model_df["agglomerative_cluster"]
            .unique()
            .tolist()
        ),
    },
    {
        "check": "missing_cluster_label_count",
        "expected": 0,
        "observed": int(
            model_df["agglomerative_cluster"]
            .isna()
            .sum()
        ),
    },
    {
        "check": "minimum_cluster_size_at_least_two",
        "expected": True,
        "observed": bool(cluster_counts.min() >= 2),
    },
]

for record in agglomerative_validation_records:
    record["passed"] = (record["expected"] == record["observed"])

agglomerative_validation_df = pd.DataFrame(agglomerative_validation_records)

agglomerative_validation_df.to_csv(AGGLOMERATIVE_VALIDATION_FILE, index=False,)

print("\nAgglomerative output validation:")
print(
    agglomerative_validation_df[
        ["check", "passed"]
    ].to_string(index=False)
)

failed_agglomerative_checks = (agglomerative_validation_df.loc[ ~agglomerative_validation_df["passed"], "check",].tolist())

if failed_agglomerative_checks:
    raise ValueError(
        "Agglomerative output validation failed: "
        f"{failed_agglomerative_checks}"
    )

CLUSTER_OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True,)

model_df.to_csv(CLUSTER_OUTPUT_FILE, index=False,)


# Cluster profile

cluster_profile = (model_df.groupby("agglomerative_cluster")[modelling_features].mean().round(2))

cluster_profile.insert(0, "district_count", cluster_counts)

cluster_profile.to_csv(CLUSTER_PROFILE_FILE)

print("\nAgglomerative cluster profile:")
print(cluster_profile.to_string())

cluster_membership_summary = model_df[["district_code", "district_name", "agglomerative_cluster"]].sort_values(by=["agglomerative_cluster", "district_name"])

cluster_membership_summary.to_csv(CLUSTER_MEMBERSHIP_FILE, index=False)

print("\nSaved agglomerative cluster membership table to:")
print(CLUSTER_MEMBERSHIP_FILE)

model_selection_summary = pd.DataFrame([
    {
        "model": "Agglomerative clustering",
        "linkage": "ward",
        "distance_metric": "euclidean",
        "tested_k_values": ", ".join(
            metrics_df["k"]
            .astype(int)
            .astype(str)
            .tolist()
        ),
        "selected_k": best_k,
        "selection_rule": (
            "Highest silhouette score among candidates with "
            "a minimum cluster size of at least two districts; "
            "lower Davies-Bouldin score used as a tie-breaker."
        ),
        "selected_silhouette_score": round(
            float(best_metrics_row["silhouette_score"]),
            4,
        ),
        "selected_davies_bouldin_score": round(
            float(best_metrics_row["davies_bouldin_score"]),
            4,
        ),
        "selected_calinski_harabasz_score": round(
            float(best_metrics_row["calinski_harabasz_score"]),
            4,
        ),
        "minimum_cluster_size": int(
            best_metrics_row["minimum_cluster_size"]
        ),
        "maximum_cluster_size": int(
            best_metrics_row["maximum_cluster_size"]
        ),
        "selection_status": (
            "Provisional until comparison with K-Means and "
            "Gaussian Mixture Models."
        ),
        "important_limitation": (
            "The analysis contains only 32 districts. Ward "
            "clustering produces descriptive hierarchical groups "
            "and does not estimate probabilistic membership."
        ),
    }
])

model_selection_summary.to_csv(
    MODEL_SELECTION_SUMMARY_FILE,
    index=False,
)

print("\nSaved Agglomerative model-selection summary to:")
print(MODEL_SELECTION_SUMMARY_FILE)

# PCA visualization only for plotting
pca = PCA(n_components=2)
pca_coordinates = pca.fit_transform(X_scaled)


pca_df = model_df[["district_code", "district_name", "agglomerative_cluster"]].copy()
pca_df["pc1"] = pca_coordinates[:, 0]
pca_df["pc2"] = pca_coordinates[:, 1]

pca_df.to_csv(PCA_OUTPUT_FILE, index=False)

print("\nSaved PCA coordinates to:")
print(PCA_OUTPUT_FILE)

plt.figure(figsize=(11, 8))

for cluster_id in sorted(pca_df["agglomerative_cluster"].unique()):
    cluster_data = pca_df[pca_df["agglomerative_cluster"] == cluster_id]
    plt.scatter(cluster_data["pc1"], cluster_data["pc2"], label=f"Cluster {cluster_id}")

for _, row in pca_df.iterrows():
    plt.text(row["pc1"], row["pc2"], row["district_name"], fontsize=7)

plt.title("Agglomerative District Clusters Using PCA Visualization")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig(PCA_FIGURE_FILE , dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved PCA cluster figure to:")
print(PCA_FIGURE_FILE)

print("\nAgglomerative clustering complete.")