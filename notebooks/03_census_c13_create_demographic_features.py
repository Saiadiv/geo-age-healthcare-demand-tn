import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/interim/census_c13_clean_initial.csv")
OUTPUT_FILE = Path("data/processed/census_c13_district_demographics.csv")

df = pd.read_csv(INPUT_FILE, dtype={"table_name": str, "state_code": str, "district_code": str, "age": str,})

print("Input shape:", df.shape)

# Keep only district rows and remove Tamil Nadu state aggregate
district_df = df[df["area_name"].str.startswith("District -")].copy()

print("\nDistrict-level age rows shape:", district_df.shape)

print("\nNumber of unique districts:")
print(district_df["area_name"].nunique())

print("\nFirst 5 district rows:")
print(district_df.head(5).to_string())

# Reset index after filtering district rows
district_df = district_df.reset_index(drop=True)

# Convert population columns from text/object to numeric
population_columns = [
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

for col in population_columns:
    district_df[col] = pd.to_numeric(district_df[col], errors="coerce")

print("\nDistrict rows after index reset:")
print(district_df.head(5).to_string())

print("\nPopulation column datatypes:")
print(district_df[population_columns].dtypes)

# Creating numeric age column for age-band calculations
district_df["age_numeric"] = district_df["age"].replace({"100+": "100"})

district_df["age_numeric"] = pd.to_numeric(district_df["age_numeric"], errors="coerce")

print("\nAge numeric check:")
print(district_df[["age", "age_numeric"]].drop_duplicates().tail(15).to_string(index=False))

# Create one row per district using all the age rows
district_totals = district_df[district_df["age"] == "All ages"].copy()
district_totals = district_totals[[
    "district_code",
    "area_name",
    "total_persons",
    "rural_persons",
    "urban_persons"
]].copy()

district_totals = district_totals.rename(
    columns={
        "total_persons": "total_population",
        "rural_persons": "rural_population",
        "urban_persons": "urban_population",
    }
)

print("\nDistrict totals preview:")
print(district_totals.head(10).to_string(index=False))

print("\nDistrict totals shape:")
print(district_totals.shape)

# Creating Age band population features

child_population = (
    district_df[(district_df["age_numeric"] >=0) & (district_df["age_numeric"] <=14)]
        .groupby(["district_code", "area_name"], as_index=False)["total_persons"].sum()
        .rename(columns={"total_persons": "child_population_0_14"})
)

young_adult_population = (
    district_df[(district_df["age_numeric"] >=15) & (district_df["age_numeric"] <=39)]
    .groupby(["district_code", "area_name"], as_index=False) ["total_persons"].sum()
    .rename(columns={"total_persons": "young_adult_population_15_39"})
)

middle_age_population = (
    district_df[(district_df["age_numeric"] >=40) &(district_df["age_numeric"] <=59)]
    .groupby(["district_code", "area_name"], as_index=False) ["total_persons"].sum()
    .rename(columns={"total_persons": "middle_age_population_40_59"})
)

elderly_population = (
    district_df[district_df["age_numeric"] >= 60]
    .groupby(["district_code", "area_name"], as_index=False)["total_persons"]
    .sum()
    .rename(columns={"total_persons": "elderly_population_60_plus"})
)

print("\nChild population preview:")
print(child_population.head().to_string(index=False))

print("\nAge-band table shapes:")
print("Child:", child_population.shape)
print("Young_Adult:", young_adult_population.shape)
print("Middle_age:", middle_age_population.shape)
print("Elderly:", elderly_population.shape)

# Merge district totals with age-band population features
demographics = district_totals.merge(child_population, on=["district_code", "area_name"], how="left")

demographics = demographics.merge(young_adult_population, on=["district_code", "area_name"], how="left")

demographics = demographics.merge(middle_age_population, on=["district_code", "area_name"], how="left")

demographics = demographics.merge(elderly_population, on=["district_code", "area_name"],how="left")

# Create combined working-age population
demographics["working_age_population_15_59"] = (demographics["young_adult_population_15_39"] + demographics["middle_age_population_40_59"])

print("\nMerged demographic feature preview:")
print(demographics.head(10).to_string(index=False))

print("\nMerged demographic feature shape:")
print(demographics.shape)

# Create Demographic share features as percentages
demographics["child_share_percent"] = (demographics["child_population_0_14"] / demographics["total_population"] * 100)

demographics["young_adult_share_percent"] = (demographics["young_adult_population_15_39"] / demographics["total_population"] * 100)

demographics["middle_age_share_percent"] = (demographics["middle_age_population_40_59"] / demographics["total_population"] * 100)

demographics["elderly_share_percent"] = (demographics["elderly_population_60_plus"] / demographics["total_population"] * 100)

demographics["rural_share_percent"] = (demographics["rural_population"] / demographics["total_population"] * 100)

demographics["urban_share_percent"] = (demographics["urban_population"] / demographics["total_population"] * 100)

print("\nDemographic share preview:")

print(
    demographics[
        [
            "district_code",
            "area_name",
            "total_population",
            "child_share_percent",
            "young_adult_share_percent",
            "middle_age_share_percent",
            "elderly_share_percent",
            "rural_share_percent",
            "urban_share_percent",
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

# Validate whether age-band populations cover the total population
demographics["age_band_population_sum"] = (
    demographics["child_population_0_14"] +
    demographics["young_adult_population_15_39"] +
    demographics["middle_age_population_40_59"] +
    demographics["elderly_population_60_plus"]
)

demographics["age_band_unassigned_population"] = (demographics["total_population"] - demographics["age_band_population_sum"])

demographics["age_band_coverage_percent"] = (demographics["age_band_population_sum"] / demographics["total_population"] * 100)

print("\nAge-band validation preview:")
print(
    demographics[
        [
            "district_code",
            "area_name",
            "total_population",
            "age_band_population_sum",
            "age_band_unassigned_population",
            "age_band_coverage_percent",
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)

# Create clean district name for chart and Reporting
demographics["district_name"] = demographics["area_name"].str.replace("District - ", "", regex=False)

demographics["district_name"] = demographics["district_name"].str.rsplit(" (", n=1).str[0].str.strip()

# Reorder columns for final processed output
final_columns = [
    "district_code",
    "district_name",
    "area_name",
    "total_population",
    "rural_population",
    "urban_population",
    "child_population_0_14",
    "young_adult_population_15_39",
    "middle_age_population_40_59",
    "elderly_population_60_plus",
    "working_age_population_15_59",
    "child_share_percent",
    "young_adult_share_percent",
    "middle_age_share_percent",
    "elderly_share_percent",
    "rural_share_percent",
    "urban_share_percent",
    "age_band_population_sum",
    "age_band_unassigned_population",
    "age_band_coverage_percent",
]

demographics = demographics[final_columns]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
demographics.to_csv(OUTPUT_FILE, index=False)

print("\nFinal demographic feature preview:")
print(demographics.head(10).round(2).to_string(index=False))

print("\nFinal demographic feature shape:")
print(demographics.shape)

print(f"\nSaved processed demographic file to: {OUTPUT_FILE}")
