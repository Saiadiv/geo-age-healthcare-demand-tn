import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/district_healthcare_demand_master_with_scores.csv")

OUTPUT_FILE = Path("data/processed/district_modelling_features.csv")
MISSING_SUMMARY_FILE = Path("reports/tables/table_modelling_feature_missing_values.csv")
IMPUTATION_LOG_FILE = Path("reports/tables/table_modelling_feature_imputation_log.csv")
MODEL_VALIDATION_FILE = Path("reports/tables/table_modelling_dataset_validation.csv")

EXPECTED_DISTRICT_COUNT = 32
EXPECTED_OUTPUT_COLUMN_COUNT = 13
master_df = pd.read_csv(INPUT_FILE, dtype={"district_code": str})

print("Master dataset with score shape:")
print(master_df.shape)

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
    "breast_cancer_screening_gap_percent"
]

# Keep District identifiers + modelling features
model_df = master_df[["district_code", "district_name"] + modelling_features].copy()

print("\nModelling dataset shape before imputation:")
print(model_df.shape)

print("\nMissing values before imputation")
missing_before = model_df[modelling_features].isna().sum()
print(missing_before.to_string())

vaccination_feature = "vaccination_gap_percent"

vaccination_suppression_mask = (master_df["full_vaccination_under_25_cases_suppressed_flag"].fillna(False).astype(bool))

vaccination_missing_mask = model_df[vaccination_feature].isna()

expected_missing_counts = pd.Series(0, index=modelling_features, dtype="int64",)

expected_missing_counts[vaccination_feature] = int(vaccination_suppression_mask.sum())

missing_summary = pd.DataFrame({
    "feature": modelling_features,
    "missing_values_before_imputation": (
        missing_before.reindex(modelling_features).values
    ),
    "expected_missing_values": (
        expected_missing_counts.reindex(modelling_features).values
    ),
})

missing_summary["missing_values_match_expected"] = (missing_summary["missing_values_before_imputation"] == missing_summary["expected_missing_values"])

MISSING_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True,)

missing_summary.to_csv(MISSING_SUMMARY_FILE, index=False,)

features_with_unexpected_missing_values = (missing_summary.loc[~missing_summary["missing_values_match_expected"], "feature",].tolist())

vaccination_missing_matches_suppression = (vaccination_missing_mask.equals(vaccination_suppression_mask))

if (features_with_unexpected_missing_values or not vaccination_missing_matches_suppression):
    raise ValueError(
        "Unexpected modelling-feature missingness. "
        f"Features: {features_with_unexpected_missing_values}; "
        "vaccination missing values match suppression flag: "
        f"{vaccination_missing_matches_suppression}"
    )

model_df["vaccination_gap_imputed_flag"] = (vaccination_missing_mask)

median_value = float(model_df.loc[~vaccination_missing_mask, vaccination_feature,].median())

affected_districts = model_df.loc[vaccination_missing_mask, "district_name",].tolist()

model_df.loc[vaccination_missing_mask, vaccination_feature,] = median_value

imputation_records = [
    {
        "feature": vaccination_feature,
        "imputation_method": "median",
        "imputed_value": median_value,
        "number_of_missing_values_imputed": int(
            vaccination_missing_mask.sum()
        ),
        "affected_districts": "; ".join(
            affected_districts
        ),
        "note": (
            "Imputation applied only to the modelling dataset; "
            "the source master dataset remains unchanged."
        ),
    }
]

missing_after = model_df[modelling_features].isna().sum()

missing_summary["missing_values_after_imputation"] = (missing_after.reindex(modelling_features).values)

missing_summary.to_csv(MISSING_SUMMARY_FILE, index=False,)

imputation_log = pd.DataFrame(imputation_records)

IMPUTATION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True,)

imputation_log.to_csv(IMPUTATION_LOG_FILE, index=False,)

if int(missing_after.sum()) > 0:
    remaining_missing = missing_after[
        missing_after > 0
    ].to_dict()

    raise ValueError(
        "Missing values remain after imputation: "
        f"{remaining_missing}"
    )


print("\nImputation log:")
print(imputation_log.to_string(index=False))

print("\nSaved imputation log to:")
print(IMPUTATION_LOG_FILE)

expected_model_columns = ["district_code", "district_name", *modelling_features, "vaccination_gap_imputed_flag",]

invalid_modelling_value_count = int(
    (
        (model_df[modelling_features] < 0)
        | (model_df[modelling_features] > 100)
    )
    .sum()
    .sum()
)

model_validation_records = [
    {
        "check": "row_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(model_df),
    },
    {
        "check": "unique_district_name_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(
            model_df["district_name"].nunique()
        ),
    },
    {
        "check": "unique_district_code_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(
            model_df["district_code"].nunique()
        ),
    },
    {
        "check": "output_column_count",
        "expected": EXPECTED_OUTPUT_COLUMN_COUNT,
        "observed": len(model_df.columns),
    },
    {
        "check": "exact_column_order",
        "expected": expected_model_columns,
        "observed": model_df.columns.tolist(),
    },
    {
        "check": "missing_modelling_value_count",
        "expected": 0,
        "observed": int(
            model_df[modelling_features]
            .isna()
            .sum()
            .sum()
        ),
    },
    {
        "check": "invalid_modelling_value_count",
        "expected": 0,
        "observed": invalid_modelling_value_count,
    },
    {
        "check": "imputation_flag_count",
        "expected": int(
            vaccination_suppression_mask.sum()
        ),
        "observed": int(
            model_df[
                "vaccination_gap_imputed_flag"
            ].sum()
        ),
    },
]

for record in model_validation_records:
    record["passed"] = (record["expected"] == record["observed"])

model_validation_df = pd.DataFrame(model_validation_records)

MODEL_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True,)

model_validation_df.to_csv(MODEL_VALIDATION_FILE, index=False,)

print("\nFinal modelling-dataset validation:")
print(model_validation_df[["check", "passed"]].to_string(index=False))

print("\nSaved modelling-dataset validation to:")
print(MODEL_VALIDATION_FILE)

failed_model_checks = model_validation_df.loc[~model_validation_df["passed"], "check",].tolist()

if failed_model_checks:
    raise ValueError(
        "Final modelling-dataset validation failed: "
        f"{failed_model_checks}"
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
model_df.to_csv(OUTPUT_FILE, index=False)

print("\nFinal modelling dataset shape:")
print(model_df.shape)

print("\nSaved modelling dataset to:")
print(OUTPUT_FILE)
