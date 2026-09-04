import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/raw/nfhs/nfhs5_district_factsheet.xls")

OUTPUT_FILE = Path("reports/tables/nfhs5_workbook_sheet_summary.csv")

VALIDATION_OUTPUT_FILE = Path("reports/tables/nfhs5_initial_validation.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

EXPECTED_SHEET_NAMES = ["Sheet1", "Sheet2", "Sheet3"]

xls = pd.ExcelFile(INPUT_FILE, engine="xlrd")

print("NFHS sheet names:")
print(xls.sheet_names)

inspection_records = []

for sheet_name in xls.sheet_names:
    print("\n" + "=" *80)
    print("sheet:", sheet_name)

    temp_df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name, header=None, engine="xlrd")

    print("Shape:", temp_df.shape)
    print("\nFirst 12 rows:")
    print(temp_df.head(12).to_string())

    inspection_records.append({
        "sheet_name": sheet_name,
        "row_count": len(temp_df),
        "column_count": len(temp_df.columns),
        "is_empty": temp_df.empty,
    })


inspection_log = pd.DataFrame(inspection_records)
inspection_log.to_csv(OUTPUT_FILE, index=False)

print("\nInspection log")
print(inspection_log.to_string(index=False))

observed_sheet_names = xls.sheet_names

validation_records = []

def add_validation(check_name, expected, observed):
     validation_records.append({
        "check": check_name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    })

add_validation("exact_sheet_names", EXPECTED_SHEET_NAMES, observed_sheet_names,)

observed_populated_sheet_names = inspection_log.loc[~inspection_log["is_empty"], "sheet_name"].tolist()

EXPECTED_POPULATED_SHEET_NAMES = ["Sheet1"]

add_validation("populated_sheet_names", EXPECTED_POPULATED_SHEET_NAMES, observed_populated_sheet_names,)

observed_length_populated_sheets = len(observed_populated_sheet_names)

EXPECTED_LENGTH_POPULATED_SHEETS = len(EXPECTED_POPULATED_SHEET_NAMES)

add_validation("length_of_populated_sheets", EXPECTED_LENGTH_POPULATED_SHEETS, observed_length_populated_sheets,)

EXPECTED_SHEET1_SHAPE = (708, 109)

sheet1_record = inspection_log.loc[inspection_log["sheet_name"] == "Sheet1"]

if sheet1_record.empty:
     observed_sheet1_shape = None

else:
     observed_sheet1_shape = (
          int(sheet1_record["row_count"].iloc[0]),
          int(sheet1_record["column_count"].iloc[0]),
        )

add_validation("sheet1_shape", EXPECTED_SHEET1_SHAPE, observed_sheet1_shape)

validation_log = pd.DataFrame(validation_records)
validation_log.to_csv(VALIDATION_OUTPUT_FILE, index=False)

print("\nValidation log:")
print(validation_log.to_string(index=False))

if not validation_log["passed"].all():
     failed_checks = validation_log.loc[~validation_log["passed"], "check" ].to_list()

     raise ValueError(f"NFHS initial validation failed: {failed_checks}")

print("\nALL NFHS initial validation checks passed")