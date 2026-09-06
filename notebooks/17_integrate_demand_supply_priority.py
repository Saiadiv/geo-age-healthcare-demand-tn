from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


DEMAND_FILE = Path("data/processed/district_modelling_features.csv")

POPULATION_FILE = Path("data/processed/district_healthcare_demand_master_with_scores.csv")

INFRASTRUCTURE_FILE = Path("data/processed/rhs_2021_22_tamil_nadu_32_districts.csv")

CLUSTER_FILE = Path("data/processed/district_agglomerative_clusters.csv")

OUTPUT_FILE = Path("data/processed/district_demand_supply_priority.csv")

VALIDATION_FILE = Path("reports/tables/table_demand_supply_priority_validation.csv")

RANKING_FILE = Path("reports/tables/table_demand_supply_priority_ranking.csv")

SUMMARY_FILE = Path("reports/tables/table_demand_supply_index_summary.csv")

PRE_MERGE_VALIDATION_FILE = Path("reports/tables/table_demand_supply_pre_merge_validation.csv")

FIGURE_DIR = Path("reports/figures")

EXPECTED_DISTRICT_COUNT = 32
RATE_SCALE = 100_000

demand_features = [
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

facility_count_columns = [
    "sub_centres",
    "phcs",
    "chcs",
    "sub_divisional_hospitals",
    "district_hospitals",
]

facility_rate_columns = [f"{column}_per_100k_census2011_population" for column in facility_count_columns]

demand_df = pd.read_csv(DEMAND_FILE, dtype={"district_code": str},)

population_df = pd.read_csv(POPULATION_FILE, dtype={"district_code": str},)

infrastructure_df = pd.read_csv(INFRASTRUCTURE_FILE)

cluster_df = pd.read_csv(CLUSTER_FILE, dtype={"district_code": str},)

print("Demand dataset shape:", demand_df.shape)
print("Population dataset shape:", population_df.shape)
print("Infrastructure dataset shape:", infrastructure_df.shape)
print("Agglomerative cluster dataset shape:", cluster_df.shape)

input_datasets = {
    "demand": demand_df,
    "population": population_df,
    "infrastructure": infrastructure_df,
    "cluster": cluster_df,
}

pre_merge_validation_records = []

for dataset_name, dataset_df in input_datasets.items():
    pre_merge_validation_records.extend([
        {
            "check": f"{dataset_name}_row_count",
            "expected": EXPECTED_DISTRICT_COUNT,
            "observed": len(dataset_df),
        },
        {
            "check": f"{dataset_name}_unique_district_count",
            "expected": EXPECTED_DISTRICT_COUNT,
            "observed": int(
                dataset_df["district_name"].nunique()
            ),
        },
    ])

reference_district_names = set(demand_df["district_name"].dropna())

for dataset_name, dataset_df in input_datasets.items():
    observed_district_names = set(dataset_df["district_name"].dropna())

    pre_merge_validation_records.append({
        "check": f"{dataset_name}_district_names_match",
        "expected": True,
        "observed": (observed_district_names== reference_district_names),})

reference_district_codes = set(demand_df["district_code"].dropna())

for dataset_name, dataset_df in {"population": population_df, "cluster": cluster_df,}.items():
    observed_district_codes = set(dataset_df["district_code"].dropna())

    pre_merge_validation_records.append({
        "check": f"{dataset_name}_district_codes_match",
        "expected": True,
        "observed": (observed_district_codes == reference_district_codes),
    })

pre_merge_validation_df = pd.DataFrame(pre_merge_validation_records)

pre_merge_validation_df["passed"] = (pre_merge_validation_df["expected"] == pre_merge_validation_df["observed"])

PRE_MERGE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True,)

pre_merge_validation_df.to_csv(PRE_MERGE_VALIDATION_FILE, index=False,)

print("\nDemand-supply pre-merge validation:")
print(
    pre_merge_validation_df[
        ["check", "passed"]
    ].to_string(index=False)
)

failed_pre_merge_checks = pre_merge_validation_df.loc[~pre_merge_validation_df["passed"]]

if not failed_pre_merge_checks.empty:
    raise ValueError(
        "Demand-supply pre-merge validation failed: "
        f"{failed_pre_merge_checks['check'].tolist()}"
    )

integrated_df = demand_df.merge(
    population_df[["district_code", "district_name", "total_population"]],
    on=["district_code", "district_name"],
    how="left",
    validate="one_to_one",
)

integrated_df = integrated_df.merge(infrastructure_df, on="district_name", how="left",validate="one_to_one",)

integrated_df = integrated_df.merge(
    cluster_df[["district_code", "district_name", "agglomerative_cluster",]],
    on=["district_code", "district_name"],
    how="left",
    validate="one_to_one",
)

print("\nIntegrated demand-supply shape:")
print(integrated_df.shape)

invalid_population_count = int((integrated_df["total_population"].isna() | integrated_df["total_population"].le(0) ).sum())

if invalid_population_count > 0:
    raise ValueError(
        "Missing or non-positive Census population found "
        "before infrastructure-rate calculation."
    )

for count_column, rate_column in zip(facility_count_columns,facility_rate_columns,):
    integrated_df[rate_column] = (integrated_df[count_column] / integrated_df["total_population"]* RATE_SCALE)

print("\nIntegrated shape after facility rates:")
print(integrated_df.shape)

print("\nPopulation-adjusted infrastructure rates:")
print(integrated_df[["district_name", *facility_rate_columns]].head().to_string(index=False))

demand_standardized_df = pd.DataFrame(StandardScaler().fit_transform(integrated_df[demand_features]),
    columns=demand_features,
    index=integrated_df.index,
)

integrated_df["demand_component_mean"] = (demand_standardized_df.mean(axis=1))

integrated_df["healthcare_demand_index_z"] = (StandardScaler().fit_transform(integrated_df[["demand_component_mean"]])
    .ravel()
)

supply_standardized_df = pd.DataFrame(StandardScaler().fit_transform( integrated_df[facility_rate_columns]),
    columns=facility_rate_columns,
    index=integrated_df.index,
)

integrated_df["supply_component_mean"] = (supply_standardized_df.mean(axis=1))

integrated_df["infrastructure_supply_index_z"] = ( StandardScaler().fit_transform(integrated_df[["supply_component_mean"]])
    .ravel()
)

integrated_df["demand_supply_mismatch_score"] = (integrated_df["healthcare_demand_index_z"] - integrated_df["infrastructure_supply_index_z"])

print("\nDemand-supply index preview:")
print(
    integrated_df[
        [
            "district_name",
            "healthcare_demand_index_z",
            "infrastructure_supply_index_z",
            "demand_supply_mismatch_score",
        ]
    ]
    .sort_values(by="demand_supply_mismatch_score",ascending=False,)
    .head(10)
    .to_string(index=False)
)

integrated_df["priority_rank"] = (integrated_df["demand_supply_mismatch_score"].rank(ascending=False, method="min",).astype(int))

def classify_demand_supply_quadrant(row):
    demand_is_above_average = (row["healthcare_demand_index_z"] >= 0)

    supply_is_above_average = ( row["infrastructure_supply_index_z"] >= 0)

    if demand_is_above_average and not supply_is_above_average:
        return "Above-average demand / Below-average supply"

    if demand_is_above_average and supply_is_above_average:
        return "Above-average demand / Above-average supply"

    if not demand_is_above_average and not supply_is_above_average:
        return "Below-average demand / Below-average supply"

    return "Below-average demand / Above-average supply"


integrated_df["demand_supply_quadrant"] = (integrated_df.apply(classify_demand_supply_quadrant,axis=1,))

priority_ranking_df = integrated_df[
    [
        "priority_rank",
        "district_code",
        "district_name",
        "total_population",
        "agglomerative_cluster",
        "healthcare_demand_index_z",
        "infrastructure_supply_index_z",
        "demand_supply_mismatch_score",
        "demand_supply_quadrant",
    ]
].sort_values(
    by="priority_rank"
)

print("\nTop 10 demand-supply mismatch districts:")
print(priority_ranking_df.head(10).round(4).to_string(index=False))

score_columns = [
    "healthcare_demand_index_z",
    "infrastructure_supply_index_z",
    "demand_supply_mismatch_score",
]

numeric_output_columns = [
    "total_population",
    *facility_count_columns,
    *facility_rate_columns,
    *score_columns,
]

validation_records = [
    {
        "check": "row_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(integrated_df),
    },
    {
        "check": "unique_district_name_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(integrated_df["district_name"].nunique()),
    },
    {
        "check": "unique_district_code_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(integrated_df["district_code"].nunique()),
    },
    {
        "check": "missing_required_value_count",
        "expected": 0,
        "observed": int(
            integrated_df[numeric_output_columns + ["agglomerative_cluster"]]
            .isna()
            .sum()
            .sum()
        ),
    },
    {
        "check": "all_numeric_outputs_are_finite",
        "expected": True,
        "observed": bool(np.isfinite(integrated_df[numeric_output_columns].to_numpy()).all()),
    },
    {
        "check": "demand_index_mean_is_zero",
        "expected": True,
        "observed": bool(np.isclose(integrated_df["healthcare_demand_index_z"].mean(), 0, atol=1e-10,)),
    },
    {
        "check": "demand_index_sd_is_one",
        "expected": True,
        "observed": bool(
            np.isclose(
                integrated_df[
                    "healthcare_demand_index_z"
                ].std(ddof=0),
                1,
                atol=1e-10,
            )
        ),
    },
    {
        "check": "supply_index_mean_is_zero",
        "expected": True,
        "observed": bool(
            np.isclose(
                integrated_df[
                    "infrastructure_supply_index_z"
                ].mean(),
                0,
                atol=1e-10,
            )
        ),
    },
    {
        "check": "supply_index_sd_is_one",
        "expected": True,
        "observed": bool(
            np.isclose(
                integrated_df[
                    "infrastructure_supply_index_z"
                ].std(ddof=0),
                1,
                atol=1e-10,
            )
        ),
    },
    {
        "check": "mismatch_formula_reproduced",
        "expected": True,
        "observed": bool(
            np.allclose(
                integrated_df[
                    "demand_supply_mismatch_score"
                ],
                (
                    integrated_df[
                        "healthcare_demand_index_z"
                    ]
                    - integrated_df[
                        "infrastructure_supply_index_z"
                    ]
                ),
            )
        ),
    },
    {
        "check": "unique_priority_rank_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(integrated_df["priority_rank"].nunique()),
    },
    {
        "check": "minimum_priority_rank",
        "expected": 1,
        "observed": int(integrated_df["priority_rank"].min()),
    },
    {
        "check": "maximum_priority_rank",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(integrated_df["priority_rank"].max()),
    },
    {
        "check": "missing_quadrant_count",
        "expected": 0,
        "observed": int(integrated_df["demand_supply_quadrant"].isna().sum()),
    },
]

for facility in facility_count_columns:
    validation_records.append({
        "check": f"{facility}_total_preserved_after_merge",
        "expected": int(infrastructure_df[facility].sum()),
        "observed": int(integrated_df[facility].sum()),
    })

validation_df = pd.DataFrame(validation_records)

validation_df["passed"] = (validation_df["expected"]== validation_df["observed"])

VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
validation_df.to_csv(VALIDATION_FILE, index=False)

print("\nDemand-supply priority validation:")
print(validation_df[["check", "passed"]].to_string(index=False))

failed_checks = validation_df.loc[~validation_df["passed"]]

if not failed_checks.empty:
    raise ValueError(
        "Demand-supply priority validation failed: "
        f"{failed_checks['check'].tolist()}"
    )

index_summary_df = integrated_df[[*facility_rate_columns, *score_columns,]].describe().T

SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
index_summary_df.to_csv(SUMMARY_FILE)

priority_ranking_df.to_csv(RANKING_FILE, index=False,)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
integrated_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved demand-supply index summary to:")
print(SUMMARY_FILE)

print("\nSaved district priority ranking to:")
print(RANKING_FILE)

print("\nSaved integrated demand-supply dataset to:")
print(OUTPUT_FILE)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TOP_10_FIGURE_FILE = (FIGURE_DIR / "figure_top_10_demand_supply_mismatch.png")

QUADRANT_FIGURE_FILE = (FIGURE_DIR / "figure_demand_supply_priority_quadrant.png")

top_10_plot_df = (priority_ranking_df.head(10).sort_values(by="demand_supply_mismatch_score",ascending=True,))

plt.figure(figsize=(10, 7))

bars = plt.barh(
    top_10_plot_df["district_name"],
    top_10_plot_df["demand_supply_mismatch_score"],
    color="#C44E52",
)

for bar, score in zip(bars, top_10_plot_df["demand_supply_mismatch_score"],):
    plt.text(
        score + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.2f}",
        va="center",
        fontsize=9,
    )

plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 10 Districts by Demand-Supply Mismatch")
plt.xlabel("Demand index z-score-Supply index z-score")
plt.ylabel("District")
plt.tight_layout()

plt.savefig(TOP_10_FIGURE_FILE, dpi=300, bbox_inches="tight",)

plt.close()

print("\nSaved top-10 mismatch figure to:")
print(TOP_10_FIGURE_FILE)

plt.figure(figsize=(11, 8))

scatter = plt.scatter(
    integrated_df["infrastructure_supply_index_z"],
    integrated_df["healthcare_demand_index_z"],
    c=integrated_df["demand_supply_mismatch_score"],
    cmap="coolwarm",
    s=90,
    edgecolors="black",
    linewidths=0.5,
)

plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.axvline(0, color="grey", linestyle="--", linewidth=1)

top_10_district_names = set(priority_ranking_df.head(10)["district_name"])

for _, row in integrated_df.iterrows():
    if row["district_name"] in top_10_district_names:
        plt.annotate(
            row["district_name"],
            (
                row["infrastructure_supply_index_z"],
                row["healthcare_demand_index_z"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

plt.text(
    0.02,
    0.98,
    "Priority zone:\nAbove-average demand\nBelow-average supply",
    transform=plt.gca().transAxes,
    va="top",
    fontsize=9,
    color="darkred",
)

plt.colorbar(scatter,label="Demand-supply mismatch score",)

plt.title("Tamil Nadu District Demand-Supply Priority Matrix")
plt.xlabel("Infrastructure supply index (z-score)")
plt.ylabel("Healthcare demand index (z-score)")
plt.tight_layout()

plt.savefig(QUADRANT_FIGURE_FILE, dpi=300, bbox_inches="tight",)

plt.close()

print("\nSaved demand-supply quadrant figure to:")
print(QUADRANT_FIGURE_FILE)