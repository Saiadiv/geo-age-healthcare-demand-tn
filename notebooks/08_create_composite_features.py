import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/district_healthcare_demand_master.csv")
OUTPUT_FILE = Path("data/processed/district_healthcare_demand_master_with_scores.csv")

COMPOSITE_SUMMARY_FILE = Path("reports/tables/table_composite_feature_summary.csv")
COMPOSITE_DICTIONARY_FILE = Path("reports/tables/table_composite_feature_data_dictionary.csv")
COMPOSITE_VALIDATION_FILE = Path("reports/tables/table_composite_feature_validation.csv")

EXPECTED_DISTRICT_COUNT = 32
EXPECTED_OUTPUT_COLUMN_COUNT = 59

master_df = pd.read_csv(INPUT_FILE, dtype={"district_code": str})

print("Master dataset shape before composite features")
print(master_df.shape)

# Composite 1: Vaccination gap
master_df["vaccination_gap_percent"] = (100 - master_df["full_vaccination_percent"])

# Composite 2: WASH access score
master_df["wash_access_score"] = master_df[
    [
        "improved_drinking_water_percent",
        "improved_sanitation_percent",
        "clean_fuel_percent"
    ]
].mean(axis=1, skipna=False)

master_df["wash_access_gap_score"] = (
    100 - master_df["wash_access_score"]
)

# Composite 3: child health burden score
master_df["child_health_burden_score"] = master_df[
    [
        "children_stunted_percent",
        "children_underweight_percent",
        "children_anaemic_percent"
    ]
].mean(axis=1, skipna=False)


# Composite 4: NCD burden score
master_df["ncd_burden_score"] = master_df[
    [
        "women_high_blood_sugar_or_medicine_percent",
        "men_high_blood_sugar_or_medicine_percent",
        "women_elevated_bp_or_medicine_percent",
        "men_elevated_bp_or_medicine_percent"
    ]
].mean(axis=1, skipna=False)

# Composite 5: Maternal care continuity score
master_df["maternal_care_continuity_score"] = master_df[
    [
        "anc_first_trimester_percent",
        "anc_4plus_visits_percent",
        "births_attended_by_skilled_personnel_percent",
        "postnatal_care_mother_2days_percent"
    ]
].mean(axis=1, skipna=False)

master_df["maternal_care_gap_score"] = (100 - master_df["maternal_care_continuity_score"])


# Composite 6: Newborn follow-up score
master_df["newborn_followup_score"] = master_df["postnatal_care_newborn_2days_percent"]

master_df["newborn_followup_gap_score"] = (100 - master_df["newborn_followup_score"])

# Composite 7: screening gap
master_df["breast_cancer_screening_gap_percent"] = (100 - master_df["breast_cancer_screening_percent"])

# Save composite summary statistics
composite_features = [
    "vaccination_gap_percent",
    "wash_access_score",
    "wash_access_gap_score",
    "child_health_burden_score",
    "ncd_burden_score",
    "maternal_care_continuity_score",
    "maternal_care_gap_score",
    "newborn_followup_score",
    "newborn_followup_gap_score",
    "breast_cancer_screening_gap_percent",
]


validation_records = [
    {
        "check": "row_count",
        "feature": None,
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(master_df),
    },
    {
        "check": "unique_district_count",
        "feature": None,
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(master_df["district_name"].nunique()),
    },
    {
        "check": "duplicate_district_count",
        "feature": None,
        "expected": 0,
        "observed": int(master_df["district_name"].duplicated().sum()),
    },
    {
        "check": "output_column_count",
        "feature": None,
        "expected": EXPECTED_OUTPUT_COLUMN_COUNT,
        "observed": len(master_df.columns),
    },
]

expected_missing_counts = {feature: 0 for feature in composite_features}

expected_missing_counts["vaccination_gap_percent"] = int(
    master_df[
        "full_vaccination_under_25_cases_suppressed_flag"
    ].sum()
)

for feature in composite_features:
    feature_values = master_df[feature]

    observed_missing_count = int(feature_values.isna().sum())

    observed_invalid_range_count = int(((feature_values < 0)| (feature_values > 100)).sum())

    validation_records.extend([
        {
            "check": "missing_value_count",
            "feature": feature,
            "expected": expected_missing_counts[feature],
            "observed": observed_missing_count,
        },
        {
            "check": "invalid_range_count",
            "feature": feature,
            "expected": 0,
            "observed": observed_invalid_range_count,
        },
    ])

for record in validation_records:
    record["passed"] = (record["expected"] == record["observed"])

validation_df = pd.DataFrame(validation_records)

COMPOSITE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True,)

validation_df.to_csv(COMPOSITE_VALIDATION_FILE, index=False,)

print("\nComposite-feature validation:")
print(validation_df.to_string(index=False))

print("\nSaved composite-feature validation to:")
print(COMPOSITE_VALIDATION_FILE)

failed_validation_checks = validation_df.loc[~validation_df["passed"]]

if not failed_validation_checks.empty:
    failed_details = failed_validation_checks[["check", "feature"]].to_dict(orient="records")

    raise ValueError(
        "Composite-feature validation failed: "
        f"{failed_details}"
    )

composite_summary = master_df[composite_features].describe().T

COMPOSITE_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
composite_summary.to_csv(COMPOSITE_SUMMARY_FILE)

print("\nComposite feature summary:")
print(composite_summary.to_string())

print("\nSaved composite feature summary to:")
print(COMPOSITE_SUMMARY_FILE)

# Save composite feature dictionary
composite_dictionary = pd.DataFrame([
    {
        "composite_feature": "vaccination_gap_percent",
        "calculation": "100 - full_vaccination_percent",
        "interpretation": "Higher value indicates lower vaccination coverage and greater vaccination gap."
    },
    {
        "composite_feature": "wash_access_score",
        "calculation": "Average of improved drinking water, improved sanitation, and clean fuel percentages.",
        "interpretation": "Higher value indicates stronger household WASH and clean-fuel access."
    },
    {
        "composite_feature": "wash_access_gap_score",
        "calculation": "100 - wash_access_score",
        "interpretation": "Higher value indicates weaker household WASH and clean-fuel access."
    },
    {
        "composite_feature": "child_health_burden_score",
        "calculation": "Average of child stunting, child underweight, and child anaemia percentages.",
        "interpretation": "Higher value indicates greater child health and nutrition burden."
    },
    {
        "composite_feature": "ncd_burden_score",
        "calculation": "Average of male/female high blood sugar and male/female elevated blood pressure indicators.",
        "interpretation": "Higher value indicates greater NCD-related burden."
    },
    {
        "composite_feature": "maternal_care_continuity_score",
        "calculation": "Average of first-trimester ANC, 4+ ANC visits, skilled birth attendance, and postnatal care for mothers within 2 days of delivery.",
        "interpretation": "Higher value indicates stronger maternal care continuity across pregnancy, delivery, and early postnatal care."
    },
    {
        "composite_feature": "maternal_care_gap_score",
        "calculation": "100 - maternal_care_continuity_score",
        "interpretation": "Higher value indicates a larger maternal care continuity gap and greater need for maternal follow-up, gynec/obstetric outreach, and postnatal care strengthening."
    },
    {
        "composite_feature": "newborn_followup_score",
        "calculation": "Postnatal care received by newborns within 2 days of delivery.",
        "interpretation": "Higher value indicates stronger newborn postnatal follow-up within 2 days of delivery."
    },
    {
        "composite_feature": "newborn_followup_gap_score",
        "calculation": "100 - newborn_followup_score",
        "interpretation": "Higher value indicates a larger newborn follow-up gap and greater need for newborn check-up, ANM/PHC outreach, and early newborn-care strengthening."
    },
    {
        "composite_feature": "breast_cancer_screening_gap_percent",
        "calculation": "100 - breast_cancer_screening_percent",
        "interpretation": "Higher value indicates lower breast cancer screening uptake."
    }
])

COMPOSITE_DICTIONARY_FILE.parent.mkdir(parents=True, exist_ok=True)
composite_dictionary.to_csv(COMPOSITE_DICTIONARY_FILE, index=False)


print("\nSaved composite feature dictionary to:")
print(COMPOSITE_DICTIONARY_FILE)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
master_df.to_csv(OUTPUT_FILE, index=False)

print("\nMaster dataset shape after composite features:")
print(master_df.shape)

print("\nSaved master dataset with composite scores to:")
print(OUTPUT_FILE)