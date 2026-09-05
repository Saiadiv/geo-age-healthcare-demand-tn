from pathlib import Path

import pandas as pd

from sklearn.metrics import adjusted_rand_score


TABLE_DIR = Path("reports/tables")
PROCESSED_DIR = Path("data/processed")

KMEANS_SUMMARY_FILE = (TABLE_DIR / "table_kmeans_model_selection_summary.csv")

AGGLOMERATIVE_SUMMARY_FILE = (TABLE_DIR / "table_agglomerative_model_selection_summary.csv")

GMM_SUMMARY_FILE = (TABLE_DIR / "table_gmm_model_selection_summary.csv")

KMEANS_CLUSTER_FILE = (PROCESSED_DIR / "district_kmeans_clusters.csv")

AGGLOMERATIVE_CLUSTER_FILE = (PROCESSED_DIR / "district_agglomerative_clusters.csv")

GMM_CLUSTER_FILE = (PROCESSED_DIR / "district_gmm_clusters.csv")

MODEL_COMPARISON_FILE = (TABLE_DIR / "table_clustering_model_comparison.csv")

MODEL_AGREEMENT_FILE = (TABLE_DIR / "table_clustering_model_agreement.csv")

DISTRICT_COMPARISON_FILE = (PROCESSED_DIR / "district_clustering_model_comparison.csv")

FINAL_RECOMMENDATION_FILE = (TABLE_DIR / "table_final_clustering_recommendation.csv")

EXPECTED_DISTRICT_COUNT = 32

# Load model-selection summaries
kmeans_summary = pd.read_csv( KMEANS_SUMMARY_FILE).iloc[0]

agglomerative_summary = pd.read_csv(AGGLOMERATIVE_SUMMARY_FILE).iloc[0]

gmm_summary = pd.read_csv(GMM_SUMMARY_FILE).iloc[0]


# Load final district cluster assignments
kmeans_clusters = pd.read_csv(KMEANS_CLUSTER_FILE, dtype={"district_code": str},)

agglomerative_clusters = pd.read_csv(AGGLOMERATIVE_CLUSTER_FILE, dtype={"district_code": str},)

gmm_clusters = pd.read_csv(GMM_CLUSTER_FILE, dtype={"district_code": str},)


cluster_inputs = [
    (
        "K-Means",
        kmeans_clusters,
        "kmeans_cluster",
    ),
    (
        "Agglomerative",
        agglomerative_clusters,
        "agglomerative_cluster",
    ),
    (
        "GMM",
        gmm_clusters,
        "gmm_cluster",
    ),
]

for model_name, cluster_df, label_column in cluster_inputs:
    required_columns = {
        "district_code",
        "district_name",
        label_column,
    }

    missing_columns = (required_columns - set(cluster_df.columns))

    if missing_columns:
        raise ValueError(
            f"{model_name} output is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(cluster_df) != EXPECTED_DISTRICT_COUNT:
        raise ValueError(
            f"{model_name} output does not contain "
            f"{EXPECTED_DISTRICT_COUNT} rows."
        )

    if (
        cluster_df["district_code"].nunique()
        != EXPECTED_DISTRICT_COUNT
    ):
        raise ValueError(
            f"{model_name} district codes are not unique."
        )

    if cluster_df[label_column].isna().any():
        raise ValueError(
            f"{model_name} contains missing cluster labels."
        )

    # Align cluster labels for the same districts
district_comparison_df = (
    kmeans_clusters[
        [
            "district_code",
            "district_name",
            "kmeans_cluster",
        ]
    ]
    .merge(
        agglomerative_clusters[
            [
                "district_code",
                "agglomerative_cluster",
            ]
        ],
        on="district_code",
        validate="one_to_one",
    )
    .merge(
        gmm_clusters[
            [
                "district_code",
                "gmm_cluster",
            ]
        ],
        on="district_code",
        validate="one_to_one",
    )
)

if len(district_comparison_df) != EXPECTED_DISTRICT_COUNT:
    raise ValueError(
        "District cluster outputs do not align."
    )

district_comparison_df.to_csv(DISTRICT_COMPARISON_FILE, index=False,)


# Measure similarity between model groupings
agreement_df = pd.DataFrame({
    "model_pair": [
        "K-Means vs Agglomerative",
        "K-Means vs GMM",
        "Agglomerative vs GMM",
    ],
    "adjusted_rand_index": [
        adjusted_rand_score(
            district_comparison_df["kmeans_cluster"],
            district_comparison_df[
                "agglomerative_cluster"
            ],
        ),
        adjusted_rand_score(
            district_comparison_df["kmeans_cluster"],
            district_comparison_df["gmm_cluster"],
        ),
        adjusted_rand_score(
            district_comparison_df[
                "agglomerative_cluster"
            ],
            district_comparison_df["gmm_cluster"],
        ),
    ],
})

agreement_df["adjusted_rand_index"] = ( agreement_df["adjusted_rand_index"].round(4))

agreement_df.to_csv(MODEL_AGREEMENT_FILE, index=False,)

print("\nPairwise clustering agreement:")
print(agreement_df.to_string(index=False))

# Calculate final cluster-size ranges
kmeans_sizes = (kmeans_clusters["kmeans_cluster"].value_counts())

agglomerative_sizes = (agglomerative_clusters["agglomerative_cluster"].value_counts())

gmm_sizes = (gmm_clusters["gmm_cluster"].value_counts())


# Normalize common metrics into one comparison table
comparison_df = pd.DataFrame([
    {
        "model": "K-Means",
        "selected_k": int(
            kmeans_summary["selected_k"]
        ),
        "silhouette_score": float(
            kmeans_summary[
                "selected_silhouette_score"
            ]
        ),
        "davies_bouldin_score": float(
            kmeans_summary[
                "selected_davies_bouldin_score"
            ]
        ),
        "calinski_harabasz_score": float(
            kmeans_summary[
                "selected_calinski_harabasz_score"
            ]
        ),
        "minimum_cluster_size": int(
            kmeans_sizes.min()
        ),
        "maximum_cluster_size": int(
            kmeans_sizes.max()
        ),
    },
    {
        "model": "Agglomerative",
        "selected_k": int(
            agglomerative_summary["selected_k"]
        ),
        "silhouette_score": float(
            agglomerative_summary[
                "selected_silhouette_score"
            ]
        ),
        "davies_bouldin_score": float(
            agglomerative_summary[
                "selected_davies_bouldin_score"
            ]
        ),
        "calinski_harabasz_score": float(
            agglomerative_summary[
                "selected_calinski_harabasz_score"
            ]
        ),
        "minimum_cluster_size": int(
            agglomerative_sizes.min()
        ),
        "maximum_cluster_size": int(
            agglomerative_sizes.max()
        ),
    },
    {
        "model": "GMM",
        "selected_k": int(
            gmm_summary["selected_k"]
        ),
        "silhouette_score": float(
            gmm_summary["silhouette_score"]
        ),
        "davies_bouldin_score": float(
            gmm_summary["davies_bouldin_score"]
        ),
        "calinski_harabasz_score": float(
            gmm_summary["calinski_harabasz_score"]
        ),
        "minimum_cluster_size": int(
            gmm_sizes.min()
        ),
        "maximum_cluster_size": int(
            gmm_sizes.max()
        ),
    },
])

comparison_df["cluster_balance_ratio"] = (comparison_df["minimum_cluster_size"] / comparison_df["maximum_cluster_size"]).round(4)

comparison_df.to_csv(MODEL_COMPARISON_FILE, index=False,)

print("\nClustering model comparison:")
print(comparison_df.to_string(index=False))

final_recommendation_df = pd.DataFrame([
    {
        "selected_model": "Agglomerative clustering",
        "selected_k": int(
            agglomerative_summary["selected_k"]
        ),
        "primary_role": (
            "Final descriptive district segmentation."
        ),
        "decision_basis": (
            "Best Davies-Bouldin score, silhouette score close "
            "to K-Means, and an interpretable three-cluster "
            "solution that identifies a smaller high service-gap group."
        ),
        "robustness_evidence": (
            "Agglomerative and GMM assignments achieved an "
            "Adjusted Rand Index of 0.8000."
        ),
        "kmeans_role": (
            "Balanced baseline and sensitivity comparison."
        ),
        "gmm_role": (
            "Probabilistic sensitivity analysis and district-level "
            "membership-confidence assessment."
        ),
        "important_limitation": (
            "All silhouette scores are below 0.20. The clusters "
            "are exploratory planning profiles, not naturally "
            "separated or causal categories."
        ),
    }
])

final_recommendation_df.to_csv(
    FINAL_RECOMMENDATION_FILE,
    index=False,
)

print("\nFinal clustering recommendation:")
print(final_recommendation_df.to_string(index=False))

print("\nSaved final clustering recommendation to:")
print(FINAL_RECOMMENDATION_FILE)

print("\nClustering-model comparison complete.")