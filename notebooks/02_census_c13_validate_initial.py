import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/interim/census_c13_clean_initial.csv")
VALIDATION_REPORT_FILE = Path("reports/tables/table_census_initial_validation.csv")

EXPECTED_ROW_COUNT = 3399
EXPECTED_COLUMN_COUNT = 14
EXPECTED_AREA_COUNT = 33
EXPECTED_AGE_COUNT = 103

EXPECTED_COLUMNS = [
    "table_name",
    "state_code",
    "district_code",
    "area_name",
    "age",
    "total_persons",
    "total_males",
    "total_females",
    "rural_persons",
    "rural_males",
    "rural_females",
    "urban_persons",
    "urban_males",
    "urban_females",
]

df = pd.read_csv(INPUT_FILE, dtype={"table_name": str, "state_code": str, "district_code": str, "age": str,})

IDENTIFIER_COLUMNS = ["table_name", "state_code", "district_code", "area_name", "age",]

KEY_COLUMNS = ["district_code", "age"]

validation_records = []


def add_validation(check_name, expected, observed):
    validation_records.append(
        {
            "check": check_name,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
        }
    )


add_validation("row_count", EXPECTED_ROW_COUNT, df.shape[0],)

add_validation("column_count", EXPECTED_COLUMN_COUNT, df.shape[1],)

add_validation("exact_schema", EXPECTED_COLUMNS,df.columns.tolist(),)

add_validation("area_count", EXPECTED_AREA_COUNT, df["area_name"].nunique(),)

add_validation("age_count", EXPECTED_AGE_COUNT, df["age"].nunique(),)

add_validation("duplicate_district_age_keys",0, int(df.duplicated(KEY_COLUMNS).sum()),)

add_validation("missing_identifier_values", 0,int(df[IDENTIFIER_COLUMNS].isna().sum().sum()),)

add_validation("table_codes", ["C3713"], sorted(df["table_name"].dropna().unique().tolist()),)

add_validation("state_codes", ["33"], sorted(df["state_code"].dropna().unique().tolist()),)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string())

print("\nNumber of unique area names:")
print(df["area_name"].nunique())

print("\nUnique area names:")
print(df["area_name"].drop_duplicates().to_string(index=False))

print("\nNumber of unique age values:")
print(df["age"].nunique())

print("\nFirst 20 unique age values:")
print(df["age"].drop_duplicates().head(20).to_string(index=False))

print("\nLast 10 unique age values:")
print(df["age"].drop_duplicates().tail(10).to_string(index=False))

# Save structured validation report
validation_report = pd.DataFrame(validation_records)

VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
validation_report.to_csv(VALIDATION_REPORT_FILE, index=False)

print("\nValidation report:")
print(validation_report.to_string(index=False))

print("\nSaved validation report to:")
print(VALIDATION_REPORT_FILE)

# Stop the pipeline if any validation failed
failed_checks = validation_report.loc[~validation_report["passed"]]

if not failed_checks.empty:
    raise ValueError(
        "Census initial validation failed:\n"
        + failed_checks.to_string(index=False)
    )

print("\nAll Census initial validation checks passed.")
