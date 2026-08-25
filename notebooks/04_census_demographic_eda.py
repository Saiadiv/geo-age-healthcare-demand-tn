import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/processed/census_c13_district_demographics.csv")
FIGURE_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, dtype={"district_code": str})

print("Input shape:", df.shape)
print("\nColumns:")
print(df.columns.to_list())

# Summary statistics for key demographic share indicators
share_columns = [
    "child_share_percent",
    "young_adult_share_percent",
    "middle_age_share_percent",
    "elderly_share_percent",
    "rural_share_percent",
    "urban_share_percent",
    "age_band_coverage_percent",
]

summary_stats = df[share_columns].describe().round(2)

summary_stats.to_csv(TABLE_DIR / "table_eda_summary_statistics.csv")

print("\nSummary statistics:")
print(summary_stats.to_string())

# Create ranked EDA tables for report insights
top_child_share = df.sort_values("child_share_percent", ascending=False).head(10)

top_elderly_share = df.sort_values("elderly_share_percent", ascending=False).head(10)

top_rural_share = df.sort_values("rural_share_percent", ascending=False).head(10)

top_child_share.to_csv(TABLE_DIR / "table_top_10_child_share_districts.csv", index=False)
top_elderly_share.to_csv(TABLE_DIR / "table_top_10_elderly_share_districts.csv", index=False)
top_rural_share.to_csv(TABLE_DIR / "table_top_10_rural_share_districts.csv", index=False)

print("\nTop 10 districts by child share:")
print(top_child_share[["district_name", "child_share_percent", "total_population"]].round(2).to_string(index=False))

print("\nTop 10 districts by elderly share:")
print(top_elderly_share[["district_name", "elderly_share_percent", "total_population"]].round(2).to_string(index=False))

print("\nTop 10 districts by rural share:")
print(top_rural_share[["district_name", "rural_share_percent", "total_population"]].round(2).to_string(index=False))

# Create horizontal bar charts for report visualisations

def save_horizontal_bar_chart(data, value_column, title, xlabel, output_path):
    chart_data = data.sort_values(value_column, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(chart_data["district_name"], chart_data[value_column])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

save_horizontal_bar_chart(
    top_child_share,
    "child_share_percent",
    "Top 10 districts by child population share",
    "Child population share (%)",
    FIGURE_DIR / "figure_top_10_child_share.png"
)

save_horizontal_bar_chart(
    top_elderly_share,
    "elderly_share_percent",
    "Top 10 districts by elderly population share",
    "Elderly population share (%)",
    FIGURE_DIR / "figure_top_10_elderly_share.png"
)

save_horizontal_bar_chart(
    top_rural_share,
    "rural_share_percent",
    "Top 10 districts by rural population share",
    "Rural population share (%)",
    FIGURE_DIR / "figure_top_10_rural_share.png"
)

print("\nSaved EDA figures to:", FIGURE_DIR)
print("Saved EDA tables to:", TABLE_DIR)

# Create EDA insight summary table for the interim report
eda_insight_summary = pd.DataFrame(
    [
        {
            "figure_or_table": "Figure 1",
            "analysis_output": "Top 10 districts by child population share",
            "key_finding": "Krishnagiri, Viluppuram, and Dharmapuri show the highest child population shares.",
            "why_it_matters": "These districts may require stronger child-health, immunization, nutrition, and maternal-child outreach planning.",
            "linked_research_question": "RQ1 and RQ3",
        },
        {
            "figure_or_table": "Figure 2",
            "analysis_output": "Top 10 districts by elderly population share",
            "key_finding": "Erode, Namakkal, and Karur show the highest elderly population shares.",
            "why_it_matters": "These districts may require stronger chronic-care, NCD screening, geriatric-care, and diagnostic planning.",
            "linked_research_question": "RQ1 and RQ2",
        },
        {
            "figure_or_table": "Figure 3",
            "analysis_output": "Top 10 districts by rural population share",
            "key_finding": "Ariyalur, Viluppuram, and Perambalur show the highest rural population shares.",
            "why_it_matters": "These districts may require outreach-based care, PHC strengthening, mobile screening, telemedicine, and local diagnostic support.",
            "linked_research_question": "RQ1, RQ2, RQ3, and RQ4",
        },
        {
            "figure_or_table": "Table 1",
            "analysis_output": "Summary statistics for demographic share indicators",
            "key_finding": "Age-band coverage averages 99.91%, confirming that the engineered age bands capture nearly the full district population.",
            "why_it_matters": "This validates the demographic feature-engineering process and supports use of the dataset for district-level planning.",
            "linked_research_question": "RQ1",
        },
    ]
)

eda_insight_summary.to_csv(
    TABLE_DIR / "table_eda_insight_summary.csv",
    index=False
)

print("\nEDA insight summary:")
print(eda_insight_summary.to_string(index=False))
