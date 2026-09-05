import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

INPUT_FILE = Path("data/processed/district_modelling_features.csv")

TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")

METRICS_FILE = TABLE_DIR / "table_kmeans_clustering_metrics.csv"
CLUSTER_OUTPUT_FILE = Path("data/processed/district_kmeans_clusters.csv")
KMEANS_VALIDATION_FILE = (TABLE_DIR / "table_kmeans_output_validation.csv")
MODEL_SELECTION_SUMMARY_FILE = (TABLE_DIR / "table_kmeans_model_selection_summary.csv")
CLUSTER_PROFILE_FILE = TABLE_DIR / "table_kmeans_cluster_profile.csv"
PCA_OUTPUT_FILE = TABLE_DIR / "table_kmeans_pca_coordinates.csv"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

model_df = pd.read_csv(INPUT_FILE, dtype={"district_code":str},)

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

EXPECTED_DISTRICT_COUNT = 32

required_columns = [
    "district_code",
    "district_name",
    *modelling_features,
    "vaccination_gap_imputed_flag",
]

missing_required_columns = [column for column in required_columns if column not in model_df.columns]


if missing_required_columns:
    raise ValueError(
        "K-Means input is missing required columns: "
        f"{missing_required_columns}"
    )

if (
    len(model_df) != EXPECTED_DISTRICT_COUNT
    or model_df["district_name"].nunique()
    != EXPECTED_DISTRICT_COUNT
):
    raise ValueError(
        "K-Means input must contain 32 unique district rows."
    )

missing_values = model_df[modelling_features].isna().sum()

print("\nMissing values:")
print(missing_values.to_string())

if missing_values.sum() > 0:
    raise ValueError("Missing values found in modelling features. Fix before clustering.")

X = model_df[modelling_features].copy()

# Standardize features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test Multiple K values
metrics_records = []

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = kmeans.fit_predict(X_scaled)

    cluster_sizes = pd.Series(labels).value_counts()

    metrics_records.append({
        "k": k,
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X_scaled, labels),
        "inertia": kmeans.inertia_,
        "minimum_cluster_size": int(cluster_sizes.min()),
        "maximum_cluster_size": int(cluster_sizes.max()),
    })


metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(METRICS_FILE, index=False)

print("\nK-Means clustering metrics:")
print(metrics_df.to_string(index=False))

print("\nSaved K-Means metrics to:")
print(METRICS_FILE)

# Provisional K selection based on highest silhouette score

eligible_metrics_df = metrics_df.loc[metrics_df["minimum_cluster_size"] >= 2].copy()

if eligible_metrics_df.empty:
    raise ValueError("All tested K-Means solutions contain a singleton cluster.")

best_metrics_row = (eligible_metrics_df.sort_values(by=["silhouette_score", "davies_bouldin_score",], ascending=[False, True]).iloc[0])

best_k = int(best_metrics_row["k"])


# Final K-Means model using provisional best k
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10 )

model_df["kmeans_cluster"] = final_kmeans.fit_predict(X_scaled)

cluster_counts = (model_df.groupby("kmeans_cluster").size().rename("district_count"))

kmeans_validation_records = [
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
        "observed": int(model_df["kmeans_cluster"].nunique()),
    },
    {
        "check": "exact_cluster_labels",
        "expected": list(range(best_k)),
        "observed": sorted(
            model_df["kmeans_cluster"]
            .unique()
            .tolist()
        ),
    },
    {
        "check": "missing_cluster_label_count",
        "expected": 0,
        "observed": int(model_df["kmeans_cluster"].isna().sum()),
    },
    {
        "check": "minimum_cluster_size_at_least_two",
        "expected": True,
        "observed": bool(cluster_counts.min() >= 2),
    },
]

for record in kmeans_validation_records:
    record["passed"] = (record["expected"] == record["observed"])

kmeans_validation_df = pd.DataFrame(kmeans_validation_records)

kmeans_validation_df.to_csv(KMEANS_VALIDATION_FILE, index=False,)

print("\nK-Means output validation:")
print(
    kmeans_validation_df[
        ["check", "passed"]
    ].to_string(index=False)
)

failed_kmeans_checks = kmeans_validation_df.loc[~kmeans_validation_df["passed"], "check",].tolist()

if failed_kmeans_checks:
    raise ValueError(
        "K-Means output validation failed: "
        f"{failed_kmeans_checks}"
    )

CLUSTER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True,)

model_df.to_csv(CLUSTER_OUTPUT_FILE, index=False,)


# Cluster profile
cluster_profile = (model_df.groupby("kmeans_cluster")[modelling_features].mean().round(2))

cluster_profile.insert(0,"district_count",cluster_counts)

cluster_profile.to_csv(CLUSTER_PROFILE_FILE)

print("\nCluster profile:")
print(cluster_profile.to_string())

cluster_membership_summary = model_df[["district_code", "district_name", "kmeans_cluster"]].sort_values(by=["kmeans_cluster", "district_name"])

cluster_membership_summary.to_csv(TABLE_DIR / "table_kmeans_cluster_membership.csv", index=False)

print("\nSaved cluster membership table to:")
print(TABLE_DIR / "table_kmeans_cluster_membership.csv")

MODEL_SELECTION_SUMMARY_FILE = (
    TABLE_DIR / "table_kmeans_model_selection_summary.csv"
)

model_selection_summary = pd.DataFrame([
    {
        "model": "K-Means clustering",
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
        "selected_inertia": round(
            float(best_metrics_row["inertia"]),
            4,
        ),
        "minimum_cluster_size": int(
            best_metrics_row["minimum_cluster_size"]
        ),
        "maximum_cluster_size": int(
            best_metrics_row["maximum_cluster_size"]
        ),
        "selection_status": (
            "Provisional until comparison with Agglomerative "
            "clustering and Gaussian Mixture Models."
        ),
        "important_limitation": (
            "The analysis contains only 32 districts. Clusters "
            "are descriptive groupings, not causal categories, "
            "and require public-health interpretation."
        ),
    }
])

model_selection_summary.to_csv(MODEL_SELECTION_SUMMARY_FILE, index=False,)

print("\nSaved K-Means model-selection summary to:")
print(MODEL_SELECTION_SUMMARY_FILE)

print("\nSaved cluster profile to:")
print(CLUSTER_PROFILE_FILE)

# PCA visualization only for plotting
pca = PCA(n_components=2)
pca_coordinates = pca.fit_transform(X_scaled)

pca_df = model_df[["district_code", "district_name", "kmeans_cluster"]].copy()
pca_df["pc1"] = pca_coordinates[:, 0]
pca_df["pc2"] = pca_coordinates[:, 1]

pca_df.to_csv(PCA_OUTPUT_FILE, index=False)

print("\nSaved PCA coordinates to:")
print(PCA_OUTPUT_FILE)

plt.figure(figsize=(11, 8))

for cluster_id in sorted(pca_df["kmeans_cluster"].unique()):
    cluster_data = pca_df[pca_df["kmeans_cluster"] == cluster_id]
    plt.scatter(
        cluster_data["pc1"],
        cluster_data["pc2"],
        label=f"Cluster {cluster_id}"
    )

for _, row in pca_df.iterrows():
    plt.text(
        row["pc1"],
        row["pc2"],
        row["district_name"],
        fontsize=7
    )

plt.title("K-Means District Clusters Using PCA Visualization")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_kmeans_clusters_pca.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved PCA cluster figure to:")
print(FIGURE_DIR / "figure_kmeans_clusters_pca.png")

print("\nBaseline K-Means clustering complete.")