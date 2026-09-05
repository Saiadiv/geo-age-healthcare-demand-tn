import pandas as pd
from pathlib import Path


CENSUS_FILE = Path("data/processed/census_c13_district_demographics.csv")
NFHS_FILE = Path("data/interim/nfhs5_tamil_nadu_selected_indicators.csv")

OUTPUT_FILE = Path("data/processed/district_healthcare_demand_master.csv")
MATCH_CHECK_FILE = Path("reports/tables/table_census_nfhs_district_match_check.csv")
PRE_MERGE_VALIDATION_FILE = Path("reports/tables/table_census_nfhs_pre_merge_validation.csv")
MERGE_VALIDATION_FILE = Path("reports/tables/table_census_nfhs_merge_validation.csv")

EXPECTED_DISTRICT_COUNT = 32

census_df = pd.read_csv((CENSUS_FILE), dtype={"district_code": str})

nfhs_df = pd.read_csv(NFHS_FILE)

print("Census shape:")
print(census_df.shape)

print("\nNFHS shape:")
print(nfhs_df.shape)

print("\nCensus districts:")
print(census_df["district_name"].nunique())

print("\nNFHS districts:")
print(nfhs_df["district_name"].nunique())

def clean_district_name(name):
      if pd.isna(name):
            return pd.NA
      normalized_name = (str(name).strip().lower().replace("&", "and"))

      return " ".join(normalized_name.split())


census_df["district_name_clean"] = census_df["district_name"].apply(clean_district_name)
nfhs_df["district_name_clean"] = nfhs_df["district_name"].apply(clean_district_name)

pre_merge_validation_records = []

for dataset_name, dataset_df in {"census": census_df,"nfhs": nfhs_df,}.items():
    clean_district_names = dataset_df["district_name_clean"]

    checks = {
    "row_count": (
        EXPECTED_DISTRICT_COUNT,
        len(dataset_df),
    ),
    "unique_clean_district_count": (
        EXPECTED_DISTRICT_COUNT,
        int(clean_district_names.nunique(dropna=True)),
    ),
    "duplicate_clean_district_count": (
        0,
        int(clean_district_names.duplicated().sum()),
    ),
    "invalid_clean_district_name_count": (
        0,
        int(
            clean_district_names
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        ),
    ),
}

    for check_name, (expected, observed) in checks.items():
        pre_merge_validation_records.append({
            "dataset": dataset_name,
            "check": check_name,
            "expected": expected,
            "observed": observed,
            "passed": expected == observed,
        })

pre_merge_validation_df = pd.DataFrame(pre_merge_validation_records)

PRE_MERGE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True,)

pre_merge_validation_df.to_csv(PRE_MERGE_VALIDATION_FILE, index=False,)

print("\nCensus-NFHS pre-merge validation:")
print(pre_merge_validation_df.to_string(index=False))

print("\nSaved pre-merge validation to:")
print(PRE_MERGE_VALIDATION_FILE)

failed_pre_merge_checks = pre_merge_validation_df.loc[~pre_merge_validation_df["passed"]]

if not failed_pre_merge_checks.empty:
    failed_details = failed_pre_merge_checks[["dataset", "check"]].to_dict(orient="records")

    raise ValueError("Census-NFHS pre-merge validation failed: "f"{failed_details}")


match_check = census_df[["district_name", "district_name_clean"]].merge(
    nfhs_df[["district_name", "district_name_clean"]],
    on="district_name_clean",
    how="outer",
    suffixes=("_census", "_nfhs"),
    indicator=True,
    validate="one_to_one",
)

MATCH_CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
match_check.to_csv(MATCH_CHECK_FILE, index=False)

print("\nDistrict match check:")
print(match_check["_merge"].value_counts())

print("\nUnmatched districts")
print(match_check[match_check["_merge"] != "both"].to_string(index=False))

print("\nSaved district match check to:")
print(MATCH_CHECK_FILE)

unmatched_count = (match_check["_merge"] != "both").sum()

if unmatched_count > 0:
        raise ValueError("Some district names did not match. Fix district names before merging.")

master_df = census_df.merge(
      nfhs_df.drop(columns=["district_name"]),
      on="district_name_clean",
      how="left",
      validate="one_to_one",
      )

master_df = master_df.drop(columns=["district_name_clean"])

merge_validation_records = [
    {
        "check": "row_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": len(master_df),
    },
    {
        "check": "unique_district_name_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(master_df["district_name"].nunique()),
    },
    {
        "check": "unique_district_code_count",
        "expected": EXPECTED_DISTRICT_COUNT,
        "observed": int(master_df["district_code"].nunique()),
    },
    {
        "check": "duplicate_district_name_count",
        "expected": 0,
        "observed": int(master_df["district_name"].duplicated().sum()),
    },
    {
        "check": "duplicate_district_code_count",
        "expected": 0,
        "observed": int(master_df["district_code"].duplicated().sum()),
    },
    {
        "check": "missing_nfhs_state_count",
        "expected": 0,
        "observed": int(master_df["state_ut"].isna().sum()),
    },
]

for record in merge_validation_records:
    record["passed"] = (record["expected"] == record["observed"])

merge_validation_df = pd.DataFrame(merge_validation_records)

MERGE_VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True,)

merge_validation_df.to_csv(MERGE_VALIDATION_FILE, index=False,)

print("\nCensus-NFHS merged-master validation:")
print(merge_validation_df.to_string(index=False))

failed_merge_checks = merge_validation_df.loc[~merge_validation_df["passed"], "check",].tolist()

if failed_merge_checks:
    raise ValueError(
        "Census-NFHS merged-master validation failed: "
        f"{failed_merge_checks}"
    )

OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True,)

master_df.to_csv(OUTPUT_FILE,index=False,)

print("\nMaster dataset shape:")
print(master_df.shape)

print("\nSaved master dataset to:")
print(OUTPUT_FILE)