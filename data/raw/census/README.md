# Census Raw Data

## Dataset: Census C-13

Source: Office of the Registrar General and Census Commissioner, India
Dataset title: C-13: Single Year Age Returns by Residence and Sex, Tamil Nadu - 2011
URL: https://censusindia.gov.in/nada/index.php/catalog/1467
Access date: August 1, 2026
Original filename: DDW-3300C-13.XLS
SHA-256: 85bcb1d85e80fb723a270ff8e8c9617ffdb1871e04d90d40b58912c4887d7426
Coverage: Tamil Nadu aggregate and 32 Census 2011 districts

## Why We Use It

Census C-13 provides the demographic demand foundation for the project. It gives age-wise population data by residence and sex, which helps identify district-level healthcare demand patterns.


## How We Use It

The dataset is used to derive features such as:

### Population Totals
- `total_population`
- `rural_population`
- `urban_population`

### Age-Band Populations
- `child_population_0_14`
- `young_adult_population_15_39`
- `middle_age_population_40_59`
- `elderly_population_60_plus`
- `working_age_population_15_59`

### Population Shares
- `child_share_percent`
- `young_adult_share_percent`
- `middle_age_share_percent`
- `elderly_share_percent`
- `rural_share_percent`
- `urban_share_percent`

### Age-Band Coverage Checks
- `age_band_population_sum`
- `age_band_unassigned_population`
- `age_band_coverage_percent`


## What Healthcare Planning Problem It Solves

This dataset helps identify districts with different age-specific healthcare needs.

Examples:

- Higher child share may indicate child-health, vaccination, nutrition, and maternal-child outreach demand.
- Higher elderly share may indicate non-communicable disease screening, chronic-care, geriatric-care, and diagnostic support demand.
- Higher rural share may indicate outreach, mobile clinic, telemedicine, and primary-care access needs.

## Processing Outputs

- `data/interim/census_c13_clean_initial.csv` - standardized initial Census table after excluding metadata/header rows.
- `reports/tables/table_census_initial_validation.csv` -  expected, observed and pass/fail evidence for nine validation checks.
- `data/processed/census_c13_district_demographics.csv` -  district-level population totals, age bands, shares and coverage features.

## Raw Data Preservation

The raw Census XLS file is preserved unchanged and identified using its SHA-256 hash. It is excluded from Git tracking. Cleaning and feature-engineering scripts read from the raw file but write only to downstream interim and processed paths.

## Current Status
- Raw source retained locally and excluded from Git
- Initial validation completed
- The structured validation report is saved under `reports/tables`.
- Demographic features generated
- Processed output saved under `data/processed/`.
- EDA was performed and the figures are saved under `reports/figures`.
- Future mapping from the historical 32 districts to the current 38-district structure is planned
- Initial cleaned shape: 3,399 rows × 14 columns
- Areas: 33
- Age categories: 103
- Duplicate district-age keys: 0
- Missing identifier values: 0
- Initial validation checks: 9/9 passed
