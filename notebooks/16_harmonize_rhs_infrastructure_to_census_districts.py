from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/interim/rhs_2021_22_tamil_nadu_38_districts.csv")

CENSUS_REFERENCE_FILE = Path("data/processed/district_healthcare_demand_master_with_scores.csv")

OUTPUT_FILE = Path("data/processed/rhs_2021_22_tamil_nadu_32_districts.csv")

CROSSWALK_FILE = Path("reports/tables/table_rhs_38_to_32_district_crosswalk.csv")

VALIDATION_FILE = Path("reports/tables/table_rhs_32_district_harmonization_validation.csv")

EXPECTED_SOURCE_DISTRICT_COUNT = 38
EXPECTED_TARGET_DISTRICT_COUNT = 32

facility_columns = [
    "sub_centres",
    "phcs",
    "chcs",
    "sub_divisional_hospitals",
    "district_hospitals",
]

district_crosswalk = {
    # New districts consolidated into their Census-2011 parent districts
    "Chengalpattu": "Kancheepuram",
    "Kallakurichi": "Viluppuram",
    "Mayiladuthurai": "Nagapattinam",
    "Ranipet": "Vellore",
    "Thirupattur": "Vellore",
    "Tenkasi": "Tirunelveli",

    # RHS-to-Census spelling harmonization
    "Kanyakumari": "Kanniyakumari",
    "Pudukottai": "Pudukkottai",
    "Thoothukudi": "Thoothukkudi",
    "Tiruchirapalli": "Tiruchirappalli",
    "Tirupur": "Tiruppur",
    "Tiruvallur": "Thiruvallur",
    "Tiruvarur": "Thiruvarur",
    "Villupuram": "Viluppuram",
}

rhs_df = pd.read_csv(INPUT_FILE)

census_reference_df = pd.read_csv(CENSUS_REFERENCE_FILE, dtype={"district_code": str},)

print("RHS source shape:")
print(rhs_df.shape)

print("\nCensus-reference district count:")
print(census_reference_df["district_name"].nunique())

rhs_df["district_name"] = (rhs_df["source_district_name"].replace(district_crosswalk))

crosswalk_df = rhs_df[["source_district_name", "district_name"]].copy()

crosswalk_df["mapping_status"] = crosswalk_df.apply(
    lambda row: (
        "unchanged"
        if row["source_district_name"] == row["district_name"]
        else "mapped"
    ),
    axis=1,
)

print("\nRHS district crosswalk:")
print(crosswalk_df.to_string(index=False))

harmonized_df = (rhs_df.groupby("district_name", as_index=False)[facility_columns].sum())

source_district_counts = (rhs_df.groupby("district_name").size().rename("rhs_source_district_count"))

harmonized_df = harmonized_df.merge(source_district_counts, on="district_name", how="left", validate="one_to_one",)

print("\nHarmonized RHS shape:")
print(harmonized_df.shape)

print("\nHarmonized district names:")
print(harmonized_df["district_name"].to_string(index=False))

reference_district_names = set(census_reference_df["district_name"].dropna())

harmonized_district_names = set(harmonized_df["district_name"].dropna())

unmatched_harmonized_names = sorted(harmonized_district_names - reference_district_names)

missing_reference_names = sorted(reference_district_names - harmonized_district_names)

validation_records = [
    {
        "check": "source_row_count",
        "expected": EXPECTED_SOURCE_DISTRICT_COUNT,
        "observed": len(rhs_df),
    },
    {
        "check": "harmonized_row_count",
        "expected": EXPECTED_TARGET_DISTRICT_COUNT,
        "observed": len(harmonized_df),
    },
    {
        "check": "unique_harmonized_district_count",
        "expected": EXPECTED_TARGET_DISTRICT_COUNT,
        "observed": int(harmonized_df["district_name"].nunique()),
    },
    {
        "check": "duplicate_harmonized_district_count",
        "expected": 0,
        "observed": int(harmonized_df["district_name"].duplicated().sum()),
    },
    {
        "check": "unmatched_harmonized_name_count",
        "expected": 0,
        "observed": len(unmatched_harmonized_names),
    },
    {
        "check": "missing_reference_name_count",
        "expected": 0,
        "observed": len(missing_reference_names),
    },
    {
        "check": "consolidated_target_district_count",
        "expected": 5,
        "observed": int(
            (
                harmonized_df["rhs_source_district_count"] > 1
            ).sum()
        ),
    },
    {
        "check": "maximum_source_district_count",
        "expected": 3,
        "observed": int(harmonized_df["rhs_source_district_count"].max()),
    },
    {
        "check": "missing_facility_value_count",
        "expected": 0,
        "observed": int(harmonized_df[facility_columns].isna().sum().sum()),
    },
    {
        "check": "negative_facility_value_count",
        "expected": 0,
        "observed": int(
            harmonized_df[facility_columns].lt(0).sum().sum()
        ),
    },
]

for facility in facility_columns:
    validation_records.append({
        "check": f"{facility}_total_preserved",
        "expected": int(rhs_df[facility].sum()),
        "observed": int(harmonized_df[facility].sum()),
    })

validation_df = pd.DataFrame(validation_records)

validation_df["passed"] = (validation_df["expected"] == validation_df["observed"])

VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
validation_df.to_csv(VALIDATION_FILE, index=False)

print("\nRHS 38-to-32 harmonization validation:")
print(validation_df.to_string(index=False))

print("\nUnmatched harmonized names:")
print(unmatched_harmonized_names)

print("\nMissing Census-reference names:")
print(missing_reference_names)

failed_checks = validation_df.loc[~validation_df["passed"]]

if not failed_checks.empty:
    raise ValueError(
        "RHS district harmonization validation failed: "
        f"{failed_checks['check'].tolist()}"
    )

crosswalk_df = crosswalk_df.sort_values(by=["district_name", "source_district_name"])

harmonized_df = harmonized_df.sort_values(by="district_name")

CROSSWALK_FILE.parent.mkdir(parents=True, exist_ok=True)
crosswalk_df.to_csv(CROSSWALK_FILE, index=False)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
harmonized_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved RHS district crosswalk to:")
print(CROSSWALK_FILE)

print("\nSaved harmonized 32-district infrastructure data to:")
print(OUTPUT_FILE)