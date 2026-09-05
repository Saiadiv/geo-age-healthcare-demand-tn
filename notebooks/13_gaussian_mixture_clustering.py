from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

INPUT_FILE = Path("data/processed/district_modelling_features.csv")

TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")

METRICS_FILE = TABLE_DIR / "table_gmm_clustering_metrics.csv"
CLUSTER_OUTPUT_FILE = Path("data/processed/district_gmm_clusters.csv")
CLUSTER_PROFILE_FILE = TABLE_DIR / "table_gmm_cluster_profile.csv"
CLUSTER_MEMBERSHIP_FILE = TABLE_DIR / "table_gmm_cluster_membership.csv"
PROBABILITY_FILE = TABLE_DIR / "table_gmm_cluster_probabilities.csv"
PCA_OUTPUT_FILE = TABLE_DIR / "table_gmm_pca_coordinates.csv"
PCA_FIGURE_FILE = FIGURE_DIR / "figure_gmm_clusters_pca.png"
GMM_VALIDATION_FILE = (TABLE_DIR / "table_gmm_output_validation.csv")
MODEL_SELECTION_SUMMARY_FILE = (TABLE_DIR / "table_gmm_model_selection_summary.csv")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_DISTRICT_COUNT = 32
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 500
REG_COVAR = 1e-6
MINIMUM_CLUSTER_SIZE = 2
LOW_CONFIDENCE_THRESHOLD = 0.60

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

required_columns = ["district_code", "district_name", *modelling_features, "vaccination_gap_imputed_flag",]

missing_required_columns = [column for column in required_columns if column not in model_df.columns]

if missing_required_columns:
    raise ValueError(
        "GMM input is missing required columns: "
        f"{missing_required_columns}"
    )

if (
    len(model_df) != EXPECTED_DISTRICT_COUNT
    or model_df["district_name"].nunique()
    != EXPECTED_DISTRICT_COUNT
):
    raise ValueError(
        "GMM input must contain 32 unique district rows."
    )

X = model_df[modelling_features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

metrics_records = []

for k in range(2, 6):
    candidate_model = GaussianMixture(
        n_components=k,
        covariance_type=COVARIANCE_TYPE,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        reg_covar=REG_COVAR,
    )

    labels = candidate_model.fit_predict(X_scaled)

    cluster_sizes = pd.Series(labels).value_counts()
    observed_cluster_count = int(cluster_sizes.size)

    if observed_cluster_count > 1:
        candidate_silhouette = silhouette_score(
            X_scaled,
            labels,
        )
        candidate_davies_bouldin = davies_bouldin_score(
            X_scaled,
            labels,
        )
        candidate_calinski_harabasz = (
            calinski_harabasz_score(
                X_scaled,
                labels,
            )
        )
    else:
        candidate_silhouette = np.nan
        candidate_davies_bouldin = np.nan
        candidate_calinski_harabasz = np.nan

    metrics_records.append({
        "k": k,
        "covariance_type": COVARIANCE_TYPE,
        "converged": bool(candidate_model.converged_),
        "iterations": int(candidate_model.n_iter_),
        "observed_cluster_count": observed_cluster_count,
        "minimum_cluster_size": int(cluster_sizes.min()),
        "maximum_cluster_size": int(cluster_sizes.max()),
        "silhouette_score": candidate_silhouette,
        "davies_bouldin_score": (
            candidate_davies_bouldin
        ),
        "calinski_harabasz_score": (
            candidate_calinski_harabasz
        ),
        "aic": candidate_model.aic(X_scaled),
        "bic": candidate_model.bic(X_scaled),
    })

metrics_df = pd.DataFrame(metrics_records)

metrics_df.to_csv(METRICS_FILE, index=False,)

print("\nGMM clustering metrics:")
print(metrics_df.to_string(index=False))

print("\nSaved GMM metrics to:")
print(METRICS_FILE)

eligible_metrics_df = metrics_df.loc[
    metrics_df["converged"]
    & (
        metrics_df["observed_cluster_count"]
        == metrics_df["k"]
    )
    & (
        metrics_df["minimum_cluster_size"]
        >= MINIMUM_CLUSTER_SIZE
    )
    & metrics_df["silhouette_score"].notna()
].copy()

if eligible_metrics_df.empty:
    raise ValueError(
        "No GMM candidate converged with the expected "
        "component count and minimum cluster size."
    )

best_metrics_row = (
    eligible_metrics_df
    .sort_values(
        by=[
            "bic",
            "aic",
            "silhouette_score",
        ],
        ascending=[True, True, False],
    )
    .iloc[0]
)

best_k = int(best_metrics_row["k"])

print("\nProvisional best GMM k based primarily on BIC:")
print(best_k)

# Fit the selected final GMM
final_gmm_model = GaussianMixture(
    n_components=best_k,
    covariance_type=COVARIANCE_TYPE,
    random_state=RANDOM_STATE,
    n_init=N_INIT,
    max_iter=MAX_ITER,
    reg_covar=REG_COVAR,
)

final_gmm_model.fit(X_scaled)

final_labels = final_gmm_model.predict(X_scaled)
membership_probabilities = final_gmm_model.predict_proba(X_scaled)

model_df["gmm_cluster"] = final_labels

model_df["gmm_max_membership_probability"] = (membership_probabilities.max(axis=1))

sorted_probabilities = np.sort(membership_probabilities,axis=1,)

model_df["gmm_membership_margin"] = (sorted_probabilities[:, -1]- sorted_probabilities[:, -2])


model_df["gmm_low_confidence_flag"] = (model_df["gmm_max_membership_probability"] < LOW_CONFIDENCE_THRESHOLD)

# Create district-level probability table
probability_df = model_df[["district_code", "district_name", "gmm_cluster"]].copy()

for cluster_id in range(best_k):
    probability_df[f"gmm_probability_cluster_{cluster_id}"] = (
        membership_probabilities[:, cluster_id]
    )

probability_df["max_membership_probability"] = (model_df["gmm_max_membership_probability"])

probability_df["membership_margin"] = (model_df["gmm_membership_margin"])

probability_df["low_confidence_flag"] = (model_df["gmm_low_confidence_flag"])

final_cluster_sizes = (pd.Series(final_labels) .value_counts())

probability_bounds_valid = bool(((membership_probabilities >= 0) & (membership_probabilities <= 1)).all())

probability_rows_sum_to_one = bool(
    np.allclose(
        membership_probabilities.sum(axis=1),
        1.0,
        atol=1e-8,
    )
)

labels_match_highest_probability = bool(
    np.array_equal(
        final_labels,
        membership_probabilities.argmax(axis=1),
    )
)

gmm_validation_records = [
    {
        "check": "model_converged",
        "expected": True,
        "observed": bool(final_gmm_model.converged_),
    },
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
        "observed": int(model_df["gmm_cluster"].nunique()),
    },
    {
        "check": "exact_cluster_labels",
        "expected": list(range(best_k)),
        "observed": sorted(
            model_df["gmm_cluster"].unique().tolist()
        ),
    },
    {
        "check": "missing_cluster_label_count",
        "expected": 0,
        "observed": int(model_df["gmm_cluster"].isna().sum()),
    },
    {
        "check": "minimum_cluster_size_at_least_two",
        "expected": True,
        "observed": bool(
            final_cluster_sizes.min() >= MINIMUM_CLUSTER_SIZE
        ),
    },
    {
        "check": "probabilities_within_zero_and_one",
        "expected": True,
        "observed": probability_bounds_valid,
    },
    {
        "check": "probability_rows_sum_to_one",
        "expected": True,
        "observed": probability_rows_sum_to_one,
    },
    {
        "check": "labels_match_highest_probability",
        "expected": True,
        "observed": labels_match_highest_probability,
    },
]

gmm_validation_df = pd.DataFrame(gmm_validation_records)

gmm_validation_df["passed"] = (gmm_validation_df["expected"] == gmm_validation_df["observed"])

gmm_validation_df.to_csv(GMM_VALIDATION_FILE, index=False,)

print("\nGMM output validation:")
print(
    gmm_validation_df[
        ["check", "passed"]
    ].to_string(index=False)
)

failed_gmm_checks = gmm_validation_df.loc[~gmm_validation_df["passed"]]

if not failed_gmm_checks.empty:
    failed_check_names = (
        failed_gmm_checks["check"].tolist()
    )

    raise ValueError(
        "GMM output validation failed: "
        f"{failed_check_names}"
    )

model_df.to_csv(CLUSTER_OUTPUT_FILE,index=False,)

probability_df.to_csv(PROBABILITY_FILE,index=False,)

print("\nSaved district GMM cluster labels to:")
print(CLUSTER_OUTPUT_FILE)

print("\nSaved GMM membership probabilities to:")
print(PROBABILITY_FILE)

print("\nLow-confidence district count:")
print(int(model_df["gmm_low_confidence_flag"].sum()))

# Create final GMM cluster profile
cluster_counts = (model_df.groupby("gmm_cluster").size().rename("district_count"))

cluster_profile = (model_df.groupby("gmm_cluster")[modelling_features].mean().round(2))

cluster_profile.insert(0,"district_count", cluster_counts,)

cluster_profile.to_csv(CLUSTER_PROFILE_FILE,)

print("\nGMM cluster profile:")
print(cluster_profile.to_string())

print("\nSaved GMM cluster profile to:")
print(CLUSTER_PROFILE_FILE)

# Save concise district membership and confidence metadata
cluster_membership_summary = (
    model_df[
        [
            "district_code",
            "district_name",
            "gmm_cluster",
            "gmm_max_membership_probability",
            "gmm_membership_margin",
            "gmm_low_confidence_flag",
        ]
    ]
    .sort_values(
        by=["gmm_cluster", "district_name"]
    )
)

cluster_membership_summary.to_csv(CLUSTER_MEMBERSHIP_FILE, index=False,)

print("\nSaved GMM cluster membership table to:")
print(CLUSTER_MEMBERSHIP_FILE)

model_selection_summary = pd.DataFrame([
    {
        "model": "Gaussian Mixture Model",
        "covariance_type": COVARIANCE_TYPE,
        "selected_k": best_k,
        "selection_rule": (
            "Lowest BIC among converged candidates "
            "with no cluster smaller than two districts."
        ),
        "silhouette_score": round(
            float(best_metrics_row["silhouette_score"]),
            4,
        ),
        "davies_bouldin_score": round(
            float(best_metrics_row["davies_bouldin_score"]),
            4,
        ),
        "calinski_harabasz_score": round(
            float(best_metrics_row["calinski_harabasz_score"]),
            4,
        ),
        "aic": round(
            float(best_metrics_row["aic"]),
            4,
        ),
        "bic": round(
            float(best_metrics_row["bic"]),
            4,
        ),
        "minimum_cluster_size": int(
            final_cluster_sizes.min()
        ),
        "low_confidence_district_count": int(
            model_df["gmm_low_confidence_flag"].sum()
        ),
    }
])

model_selection_summary.to_csv(MODEL_SELECTION_SUMMARY_FILE,index=False,
)

print("\nSaved GMM model-selection summary to:")
print(MODEL_SELECTION_SUMMARY_FILE)

# PCA is used only to visualize the final GMM clusters
pca = PCA(n_components=2)

pca_coordinates = pca.fit_transform(X_scaled)

pca_df = model_df[
    [
        "district_code",
        "district_name",
        "gmm_cluster",
        "gmm_max_membership_probability",
        "gmm_low_confidence_flag",
    ]
].copy()

pca_df["pc1"] = pca_coordinates[:, 0]
pca_df["pc2"] = pca_coordinates[:, 1]

pca_df.to_csv(PCA_OUTPUT_FILE,index=False,)

print("\nSaved GMM PCA coordinates to:")
print(PCA_OUTPUT_FILE)

plt.figure(figsize=(11, 8))

for cluster_id in sorted(pca_df["gmm_cluster"].unique()):
    cluster_data = pca_df.loc[pca_df["gmm_cluster"] == cluster_id]

    plt.scatter(
        cluster_data["pc1"],
        cluster_data["pc2"],
        label=f"Cluster {cluster_id}",
        alpha=0.8,
    )

for _, row in pca_df.iterrows():
    plt.text(
        row["pc1"],
        row["pc2"],
        row["district_name"],
        fontsize=7,
    )

plt.title("Gaussian Mixture District Clusters Using PCA Visualization")

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")

plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")

plt.legend()
plt.tight_layout()

plt.savefig(PCA_FIGURE_FILE, dpi=300, bbox_inches="tight",)

plt.close()

print("\nSaved GMM PCA cluster figure to:")
print(PCA_FIGURE_FILE)

print("\nGaussian Mixture clustering complete.")