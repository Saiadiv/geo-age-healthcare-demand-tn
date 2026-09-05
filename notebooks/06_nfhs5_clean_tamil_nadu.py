import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/raw/nfhs/nfhs5_district_factsheet.xls")

df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1", header=0, engine="xlrd")

df.columns = df.columns.astype(str).str.strip()

print("NFHS Sheet1 shape:", df.shape)

tn_df = df.loc[df["State/UT"] == "Tamil Nadu"].copy()

print("\nTamil Nadu NFHS shape:")
print(tn_df.shape)

# Final NFHS-5 Tamil Nadu feature selection for MVP

SELECTED_INDICATORS_OUTPUT_FILE = Path("data/interim/nfhs5_tamil_nadu_selected_indicators.csv")
DATA_DICTIONARY_FILE = Path("reports/tables/table_nfhs5_selected_indicators_data_dictionary.csv")

selected_columns = {
    # Identifiers and sample-size metadata
    "District Names": "district_name",
    "State/UT": "state_ut",
    "Number of Households surveyed": "nfhs_households_surveyed",
    "Number of Women age 15-49 years interviewed": "nfhs_women_interviewed",
    "Number of Men age 15-54 years interviewed": "nfhs_men_interviewed",

    # Access and social determinants
    "Population living in households with an improved drinking-water source1 (%)": "improved_drinking_water_percent",
    "Population living in households that use an improved sanitation facility2 (%)": "improved_sanitation_percent",
    "Households using clean fuel for cooking3 (%)": "clean_fuel_percent",
    "Households with any usual member covered under a health insurance/financing scheme (%)": "health_insurance_percent",
    "Women (age 15-49)  with 10 or more years of schooling (%)": "women_10plus_schooling_percent",

    # Maternal and newborn care continuity indicators
    "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)":
    "teenage_motherhood_percent",
    "Mothers who had an antenatal check-up in the first trimester  (for last birth in the 5 years before the survey) (%)":
    "anc_first_trimester_percent",
    "Mothers who received postnatal care from a doctor/nurse/LHV/ANM/midwife/other health personnel within 2 days of delivery (for last birth in the 5 years before the survey) (%)":
    "postnatal_care_mother_2days_percent",
    "Children who received postnatal care from a doctor/nurse/LHV/ANM/midwife/ other health personnel within 2 days of delivery (for last birth in the 5 years before the survey) (%)":
    "postnatal_care_newborn_2days_percent",
    "Births attended by skilled health personnel (in the 5 years before the survey)10 (%)":
    "births_attended_by_skilled_personnel_percent",

    # Maternal-child health and child burden
    "Mothers who had at least 4 antenatal care visits  (for last birth in the 5 years before the survey) (%)":
    "anc_4plus_visits_percent",
    "Institutional births (in the 5 years before the survey) (%)":
    "institutional_births_percent",
    "Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall11 (%)":
    "full_vaccination_percent",
    "Children under 5 years who are stunted (height-for-age)18 (%)":
    "children_stunted_percent",
    "Children under 5 years who are underweight (weight-for-age)18 (%)":
    "children_underweight_percent",
    "Children age 6-59 months who are anaemic (<11.0 g/dl)22 (%)":
    "children_anaemic_percent",

    # Adult health, NCD burden, and screening
    "Women (age 15-49 years) who are overweight or obese (BMI ≥25.0 kg/m2)21 (%)": "women_overweight_obese_percent",
    "All women age 15-49 years who are anaemic22 (%)": "women_anaemic_percent",
    "Women age 15 years and above wih high or very high (>140 mg/dl) Blood sugar level or taking medicine to control blood sugar level23 (%)": "women_high_blood_sugar_or_medicine_percent",
    "Men age 15 years and above wih high or very high (>140 mg/dl) Blood sugar level  or taking medicine to control blood sugar level23 (%)": "men_high_blood_sugar_or_medicine_percent",
    "Women age 15 years and above wih Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%)": "women_elevated_bp_or_medicine_percent",
    "Men age 15 years and above wih Elevated blood pressure (Systolic ≥140 mm of Hg and/or Diastolic ≥90 mm of Hg) or taking medicine to control blood pressure (%)": "men_elevated_bp_or_medicine_percent",
    "Women (age 30-49 years) Ever undergone a breast examination for breast cancer (%)": "breast_cancer_screening_percent",
}

missing_columns = [
    col for col in selected_columns.keys()
    if col not in tn_df.columns
]

if missing_columns:
    print("\nMissing selected columns:")
    for col in missing_columns:
        print("-", col)
    raise  ValueError("Some selected NFHS columns were not found. Check exact column names.")

nfhs_selected = tn_df[list(selected_columns.keys())].copy()
nfhs_selected = nfhs_selected.rename(columns=selected_columns)

nfhs_selected["district_name"] = (nfhs_selected["district_name"].astype("string").str.strip())

nfhs_selected["state_ut"] = (nfhs_selected["state_ut"].astype("string").str.strip())

full_vaccination_source_values = (
    nfhs_selected["full_vaccination_percent"]
    .astype("string")
    .str.strip()
)

numeric_columns = [
    col for col in nfhs_selected.columns
    if col not in ["district_name", "state_ut"]
]

for col in numeric_columns:
    nfhs_selected[col] = (
        nfhs_selected[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    nfhs_selected[col] = pd.to_numeric(nfhs_selected[col], errors="coerce")


# Decode NFHS reliability markers for full vaccination.
# Parenthesized estimates are stored as negative values and are
# based on 25-49 unweighted cases.
# An asterisk indicates a suppressed estimate based on fewer
# than 25 unweighted cases.

nfhs_selected["full_vaccination_25_49_cases_flag"] = nfhs_selected["full_vaccination_percent"].lt(0)

nfhs_selected["full_vaccination_under_25_cases_suppressed_flag"] = (full_vaccination_source_values.eq("*").fillna(False))

nfhs_selected["full_vaccination_percent"] = (nfhs_selected["full_vaccination_percent"].abs())

VACCINATION_QUALITY_FILE = Path("reports/tables/table_nfhs5_full_vaccination_quality_flags.csv")

vaccination_quality_log = pd.DataFrame({
    "district_name": nfhs_selected["district_name"],
    "source_full_vaccination_value": (
        full_vaccination_source_values
    ),
    "cleaned_full_vaccination_percent": (
        nfhs_selected["full_vaccination_percent"]
    ),
    "based_on_25_49_unweighted_cases": (
        nfhs_selected[
            "full_vaccination_25_49_cases_flag"
        ]
    ),
    "suppressed_fewer_than_25_unweighted_cases": (
        nfhs_selected[
            "full_vaccination_under_25_cases_suppressed_flag"
        ]
    ),
})

vaccination_quality_log["source_quality_status"] = ("standard_estimate")

vaccination_quality_log.loc[vaccination_quality_log["based_on_25_49_unweighted_cases"],"source_quality_status"] = "based_on_25_49_unweighted_cases"

vaccination_quality_log.loc[vaccination_quality_log["suppressed_fewer_than_25_unweighted_cases"],"source_quality_status"] = "suppressed_fewer_than_25_unweighted_cases"

VACCINATION_QUALITY_FILE.parent.mkdir(parents=True, exist_ok=True)

vaccination_quality_log.to_csv(VACCINATION_QUALITY_FILE, index=False)

print("\nFull-vaccination estimates based on 25-49 cases:")
print(nfhs_selected["full_vaccination_25_49_cases_flag"].sum())

print("\nSuppressed full-vaccination estimates:")
print(nfhs_selected["full_vaccination_under_25_cases_suppressed_flag"].sum())

print("\nSaved vaccination quality log to:")
print(VACCINATION_QUALITY_FILE)

EXPECTED_TN_DISTRICT_COUNT = 32
EXPECTED_STATE_VALUES = ["Tamil Nadu"]

observed_row_count = len(nfhs_selected)

observed_unique_district_count = (nfhs_selected["district_name"].nunique(dropna=True))

observed_duplicate_district_count = int(nfhs_selected["district_name"].duplicated().sum())

observed_invalid_district_name_count = int(nfhs_selected["district_name"].fillna("").str.strip().eq("").sum())

observed_state_values = sorted(nfhs_selected["state_ut"].dropna().unique() .tolist())

structural_validation_records = [
    {
        "check": "row_count",
        "expected": EXPECTED_TN_DISTRICT_COUNT,
        "observed": observed_row_count,
        "passed": observed_row_count == EXPECTED_TN_DISTRICT_COUNT,
    },
    {
        "check": "unique_district_count",
        "expected": EXPECTED_TN_DISTRICT_COUNT,
        "observed": observed_unique_district_count,
        "passed": (
            observed_unique_district_count
            == EXPECTED_TN_DISTRICT_COUNT
        ),
    },
    {
        "check": "duplicate_district_count",
        "expected": 0,
        "observed": observed_duplicate_district_count,
        "passed": observed_duplicate_district_count == 0,
    },
    {
        "check": "invalid_district_name_count",
        "expected": 0,
        "observed": observed_invalid_district_name_count,
        "passed": observed_invalid_district_name_count == 0,
    },
    {
        "check": "state_values",
        "expected": EXPECTED_STATE_VALUES,
        "observed": observed_state_values,
        "passed": observed_state_values == EXPECTED_STATE_VALUES,
    },
]

structural_validation_df = pd.DataFrame(structural_validation_records)

STRUCTURAL_VALIDATION_FILE = Path("reports/tables/table_nfhs5_structural_validation.csv")

STRUCTURAL_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)

structural_validation_df.to_csv(STRUCTURAL_VALIDATION_FILE, index=False)

print("\nNFHS structural validation:")
print(structural_validation_df.to_string(index=False))

failed_structural_checks = structural_validation_df.loc[~structural_validation_df["passed"], "check"].to_list()

if failed_structural_checks:
    raise ValueError(
        "NFHS structural validation failed: "
        f"{failed_structural_checks}"
)

# Validate percentage columns
percentage_columns = [col for col in nfhs_selected.columns if col.endswith("_percent")]


expected_missing_counts = {
    "full_vaccination_percent": int(
        nfhs_selected[
            "full_vaccination_under_25_cases_suppressed_flag"
        ].sum()
    )
}

validation_records = []

for col in percentage_columns:
    missing_count = int(nfhs_selected[col].isna().sum())

    expected_missing_count = (expected_missing_counts.get(col, 0))

    invalid_mask = ((nfhs_selected[col] < 0) | (nfhs_selected[col] > 100))

    validation_records.append({
        "column_name": col,
        "missing_values": missing_count,
        "expected_missing_values": expected_missing_count,
        "missing_values_match_expected": (
            missing_count == expected_missing_count
        ),
        "invalid_out_of_range_values": int(
            invalid_mask.sum()
        ),
        "minimum_value": nfhs_selected[col].min(),
        "maximum_value": nfhs_selected[col].max(),
    })

validation_df = pd.DataFrame(validation_records)

VALIDATION_FILE = Path("reports/tables/table_nfhs5_selected_indicators_validation.csv")
VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
validation_df.to_csv(VALIDATION_FILE, index=False)

print("\nNFHS selected-indicator validation:")
print(validation_df.to_string(index=False))

print("\nSaved NFHS validation table to:")
print(VALIDATION_FILE)


columns_with_unexpected_missing_values = (validation_df.loc[~validation_df["missing_values_match_expected"], "column_name"].tolist())

columns_with_out_of_range_values = (validation_df.loc[validation_df["invalid_out_of_range_values"] > 0, "column_name"].to_list())

if (columns_with_unexpected_missing_values or columns_with_out_of_range_values):
    raise ValueError(
        "NFHS percentage validation failed. "
        f"Missing values: {columns_with_unexpected_missing_values}; "
        f"out-of-range values: "
        f"{columns_with_out_of_range_values}"
        )


SELECTED_INDICATORS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
nfhs_selected.to_csv(SELECTED_INDICATORS_OUTPUT_FILE, index=False)

print("\nSelected NFHS Tamil Nadu shape:")
print(nfhs_selected.shape)

print("\nSaved selected NFHS file to:")
print(SELECTED_INDICATORS_OUTPUT_FILE)

data_dictionary = pd.DataFrame({
    "source_column": list(selected_columns.keys()),
    "renamed_column": list(selected_columns.values())
})

derived_flag_dictionary = pd.DataFrame({
    "source_column": [
        (
            "Derived from a negative/parenthesized "
            "full_vaccination_percent source value"
        ),
        (
            "Derived from the '*' full-vaccination "
            "source marker"
        ),
    ],
    "renamed_column": [
        "full_vaccination_25_49_cases_flag",
        (
            "full_vaccination_under_25_cases_"
            "suppressed_flag"
        ),
    ],
})

data_dictionary = pd.concat(
    [
        data_dictionary,
        derived_flag_dictionary,
    ],
    ignore_index=True
)

DATA_DICTIONARY_FILE.parent.mkdir(parents=True, exist_ok=True)
data_dictionary.to_csv(DATA_DICTIONARY_FILE, index=False)

print("\nSaved NFHS selected-indicator data dictionary to:")
print(DATA_DICTIONARY_FILE)