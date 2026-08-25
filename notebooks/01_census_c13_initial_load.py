import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Census C-13 Initial Load
# Why:
#   Census C-13 gives age-wise population data for Tamil Nadu.
# How:
#   We read the raw Excel file, remove title/header rows, and assign clean column names.
# What problem it solves:
#   It prepares the demographic demand layer for child, adult, elderly, rural, and urban healthcare planning.
# --------------------------------------------------

RAW_FILE = Path("data/raw/census/DDW-3300C-13.XLS")
OUTPUT_FILE = Path("data/interim/census_c13_clean_initial.csv")

# Read raw excel file without assuming first row is the header
raw_df = pd.read_excel(RAW_FILE, sheet_name="C-13", header=None, engine="xlrd")

EXPECTED_COLUMN_COUNT = 14

if raw_df.shape[1] != EXPECTED_COLUMN_COUNT:
    raise ValueError(f"Unexpected Census column count: "
        f"expected {EXPECTED_COLUMN_COUNT}, found {raw_df.shape[1]}")

print("Raw Shape:", raw_df.shape)
print("\nRaw preview:")
print(raw_df.head(12).to_string())

# Actual data starts from row index 7
census_c13 = raw_df.iloc[7:, :].copy()

# Assign clean column names
census_c13.columns = [
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

# Reset row index after removing title/header rows
census_c13 = census_c13.reset_index(drop=True)

if census_c13.empty:
      raise ValueError("Census dataset is empty after removing header rows.")

first_record = census_c13.iloc[0]

if (first_record["table_name"] != "C3713" or first_record["age"] != "All ages"):
     raise ValueError("Unexpected Census layout: row index 7 is not the expected first data record.")

print("\nCleaned preview:")
print(census_c13.head(10).to_string())

print("\nCleaned shape:", census_c13.shape)

# Save initial cleaned version
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
census_c13.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved cleaned initial file to: {OUTPUT_FILE}")
