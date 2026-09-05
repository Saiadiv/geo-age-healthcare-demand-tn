from pathlib import Path

import pandas as pd
import pdfplumber


INPUT_FILE = Path("data/raw/infrastructure/rhs_2021_22.pdf")

OUTPUT_FILE = Path("data/interim/rhs_2021_22_tamil_nadu_38_districts.csv")

VALIDATION_FILE = Path("reports/tables/table_rhs_2021_22_initial_validation.csv")

SOURCE_PDF_PAGES = [124, 125]

EXPECTED_DISTRICT_COUNT = 38

EXPECTED_FACILITY_TOTALS = {
    "sub_centres": 8713,
    "phcs": 1886,
    "chcs": 400,
    "sub_divisional_hospitals": 282,
    "district_hospitals": 20,
}

facility_columns = ["sub_centres", "phcs", "chcs", "sub_divisional_hospitals", "district_hospitals"]

extraction_records = []
collecting_tamil_nadu_rows = False

with pdfplumber.open(INPUT_FILE) as pdf:
    for pdf_page_number in SOURCE_PDF_PAGES:
        page = pdf.pages[pdf_page_number -1]
        table = page.extract_table()

        if table is None:
            raise ValueError(f"No table found on PDF page {pdf_page_number}.")

        for row in table[3:]:
            state_name = " ".join((row[1] or "").split())
            source_district_name = " ".join((row[2] or "").split())

            if state_name == "Tamil Nadu":
                collecting_tamil_nadu_rows = True

            if not collecting_tamil_nadu_rows:
                continue

            if source_district_name.startswith("Total Districts"):
                collecting_tamil_nadu_rows = False
                break

            if not source_district_name:
                continue

            facility_values = [
                int((value or "0").replace(",", "").strip())
                for value in row[3:8]
            ]

            extraction_records.append({
                "source_district_name": source_district_name,
                **dict(zip(facility_columns, facility_values)),
                "source_pdf_page": pdf_page_number,
            })

rhs_df = pd.DataFrame(extraction_records)

print("Extracted Tamil Nadu RHS shape:")
print(rhs_df.shape)

print("\nExtracted district names:")
print(rhs_df["source_district_name"].to_string(index=False))

print("\nExtracted columns:")
print(rhs_df.columns.tolist())

print("\nExtracted facility totals:")
print(rhs_df[facility_columns].sum().to_string())

validation_records = [
    {
        "check": "row_c ount",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(rhs_df),
    },
    {
        "check": "unique_district_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(rhs_df["source_district_name"].nunique()),
    },
    {
        "check": "duplicate_district_count",
        "expected": 0,
        "observed": int(rhs_df["source_district_name"].duplicated().sum()),
    },
    {
        "check": "invalid_district_name_count",
        "expected": 0,
        "observed": int(
            rhs_df["source_district_name"]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        ),
    },
    {
        "check": "missing_facility_value_count",
        "expected": 0,
        "observed": int(rhs_df[facility_columns].isna().sum().sum()),
    },
    {
        "check": "negative_facility_value_count",
        "expected": 0,
        "observed": int(rhs_df[facility_columns].lt(0).sum().sum()),
    },
]

for facility, expected_total in EXPECTED_FACILITY_TOTALS.items():
    validation_records.append({
        "check": f"{facility}_total",
        "expected": expected_total,
        "observed": int(rhs_df[facility].sum())

    })

validation_df = pd.DataFrame(validation_records)

validation_df["passed"] = (validation_df["expected"] == validation_df["observed"])

VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
validation_df.to_csv(VALIDATION_FILE, index=False)

print("\nRHS initial-extraction validation:")
print(validation_df.to_string(index=False))

print("\nSaved RHS validation table to:")
print(VALIDATION_FILE)

failed_checks = validation_df.loc[~validation_df["passed"]]

if not failed_checks.empty:
    raise ValueError(
        "RHS initial-extraction validation failed: "
        f"{failed_checks['check'].tolist()}"
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
rhs_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved extracted Tamil Nadu infrastructure data to:")
print(OUTPUT_FILE)
