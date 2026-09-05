import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path("data/processed/district_healthcare_demand_master_with_scores.csv")

TABLE_DIR = Path("reports/tables")
FIGURE_DIR = Path("reports/figures")


TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EDA_VALIDATION_FILE = (TABLE_DIR / "table_integrated_eda_input_validation.csv")

EXPECTED_DISTRICT_COUNT = 32
EXPECTED_INPUT_COLUMN_COUNT = 59

df = pd.read_csv(INPUT_FILE, dtype={"district_code": str})

print("Integrated master dataset shape:")
print(df.shape)

print("\nDistrict count:")
print(df["district_name"].nunique())

# Key EDA features
eda_features = [
    "child_share_percent",
    "elderly_share_percent",
    "rural_share_percent",
    "vaccination_gap_percent",
    "wash_access_gap_score",
    "child_health_burden_score",
    "maternal_care_gap_score",
    "newborn_followup_gap_score",
    "ncd_burden_score",
    "breast_cancer_screening_gap_percent",
    "health_insurance_percent",
    "women_10plus_schooling_percent",
]


EXPECTED_DISTRICT_COUNT = 32

required_columns = ["district_name", "total_population", *eda_features,]

missing_required_columns = [column for column in required_columns if column not in df.columns]

if missing_required_columns:
    raise ValueError(
        "Integrated EDA input is missing required columns: "
        f"{missing_required_columns}"
    )

observed_district_count = int(df["district_name"].nunique())

if (
    len(df) != EXPECTED_DISTRICT_COUNT
    or observed_district_count != EXPECTED_DISTRICT_COUNT
):
    raise ValueError(
        "Integrated EDA expected 32 unique district rows; "
        f"observed rows={len(df)}, "
        f"unique districts={observed_district_count}"
    )


print("\nMissing values in EDA features:")
print(df[eda_features].isna().sum().to_string())

# Summary statistics
summary_stats = df[eda_features].describe().T
summary_stats.to_csv(TABLE_DIR / "table_integrated_eda_summary_statistics.csv")

print("\nIntegrated EDA summary statistics:")
print(summary_stats.to_string())

# Ranking helper function
def save_top_10_table(dataframe, feature, output_filename):
    top_10 = dataframe[
        ["district_name", "total_population", feature]
    ].sort_values(
        by=feature,
        ascending=False
    ).head(10)

    top_10.to_csv(TABLE_DIR / output_filename, index=False)

    print("\nTop 10 districts by", feature)
    print(top_10.to_string(index=False))

    return top_10


top_child_burden = save_top_10_table(
    df,
    "child_health_burden_score",
    "table_top_10_child_health_burden_districts.csv"
)

top_ncd_burden = save_top_10_table(
    df,
    "ncd_burden_score",
    "table_top_10_ncd_burden_districts.csv"
)

top_wash_gap = save_top_10_table(
    df,
    "wash_access_gap_score",
    "table_top_10_wash_access_gap_districts.csv"
)

top_breast_cancer_screening_gap = save_top_10_table(
    df,
    "breast_cancer_screening_gap_percent",
    "table_top_10_breast_cancer_screening_gap_districts.csv",
)

top_maternity_care_gap = save_top_10_table(
    df,
    "maternal_care_gap_score",
    "table_top_10_maternal_care_gap_districts.csv"
)

top_newborn_followup_gap = save_top_10_table(
    df,
    "newborn_followup_gap_score",
    "table_top_10_newborn_followup_gap_districts.csv"
)


top_vaccination_gap = save_top_10_table(
    df.dropna(subset=["vaccination_gap_percent"]),
    "vaccination_gap_percent",
    "table_top_10_vaccination_gap_districts.csv"
)

# Correlation matrix
correlation_matrix = df[eda_features].corr(numeric_only=True)
correlation_matrix.to_csv(TABLE_DIR / "table_integrated_eda_correlation_matrix.csv")

print("\nCorrelation matrix saved to:")
print(TABLE_DIR / "table_integrated_eda_correlation_matrix.csv")


# Chart helper function
def save_horizontal_bar_chart(dataframe, feature, title, xlabel, output_filename):
    plot_df = dataframe.sort_values(by=feature, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["district_name"], plot_df[feature])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("District")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_filename, dpi=300)
    plt.close()

    print("Saved figure:", FIGURE_DIR / output_filename)


save_horizontal_bar_chart(
    top_child_burden,
    "child_health_burden_score",
    "Top 10 Districts by Child Health Burden Score",
    "Child health burden score",
    "figure_top_10_child_health_burden.png"
)

save_horizontal_bar_chart(
    top_ncd_burden,
    "ncd_burden_score",
    "Top 10 Districts by NCD Burden Score",
    "NCD burden score",
    "figure_top_10_ncd_burden.png"
)

save_horizontal_bar_chart(
    top_wash_gap,
    "wash_access_gap_score",
    "Top 10 Districts by WASH Access Gap Score",
    "WASH access gap score",
    "figure_top_10_wash_access_gap.png"
)

save_horizontal_bar_chart(
    top_breast_cancer_screening_gap,
    "breast_cancer_screening_gap_percent",
    "Top 10 Districts by Breast Cancer Screening Gap",
    "Breast cancer screening gap (%)",
    "figure_top_10_breast_cancer_screening_gap.png",
)

save_horizontal_bar_chart(
    top_maternity_care_gap,
    "maternal_care_gap_score",
    "Top 10 Districts by Maternal Care Gap",
    "Maternal care gap score",
    "figure_top_10_maternal_care_gap.png",
)

save_horizontal_bar_chart(
    top_newborn_followup_gap,
    "newborn_followup_gap_score",
    "Top 10 Districts by Newborn Follow-up Gap",
    "Newborn follow-up gap score",
    "figure_top_10_newborn_followup_gap.png",
)

save_horizontal_bar_chart(
    top_vaccination_gap,
    "vaccination_gap_percent",
    "Top 10 Districts by Vaccination Gap",
    "Vaccination gap percent",
    "figure_top_10_vaccination_gap.png"
)


# Scatter plot helper
def save_scatter_plot(dataframe, x_col, y_col, title, xlabel, ylabel, output_filename):
    plot_df = dataframe.dropna(subset=[x_col, y_col])

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df[x_col], plot_df[y_col])

    for _, row in plot_df.iterrows():
        plt.text(row[x_col], row[y_col], row["district_name"], fontsize=7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_filename, dpi=300)
    plt.close()

    print("Saved figure:", FIGURE_DIR / output_filename)


save_scatter_plot(
    df,
    "child_share_percent",
    "child_health_burden_score",
    "Child Share vs Child Health Burden",
    "Child population share (%)",
    "Child health burden score",
    "figure_scatter_child_share_vs_child_health_burden.png"
)

save_scatter_plot(
    df,
    "elderly_share_percent",
    "ncd_burden_score",
    "Elderly Share vs NCD Burden",
    "Elderly population share (%)",
    "NCD burden score",
    "figure_scatter_elderly_share_vs_ncd_burden.png"
)

save_scatter_plot(
    df,
    "rural_share_percent",
    "wash_access_gap_score",
    "Rural Share vs WASH Access Gap",
    "Rural population share (%)",
    "WASH access gap score",
    "figure_scatter_rural_share_vs_wash_gap.png"
)

save_scatter_plot(
    df,
    "women_10plus_schooling_percent",
    "breast_cancer_screening_gap_percent",
    "Women's Schooling vs Breast Cancer Screening Gap",
    "Women with 10+ years schooling (%)",
    "Breast cancer screening gap (%)",
    "figure_scatter_women_schooling_vs_breast_cancer_screening_gap.png"
)

# Minimum sample size by research question

# Observed district coverage by research question

district_coverage_by_rq = pd.DataFrame([
    {
        "research_question": "RQ1",
        "analysis_focus": (
            "Demographic demand clusters using Census age "
            "and rural-urban features"
        ),
        "analysis_unit": "District",
        "observed_district_count": int(
            df["district_name"].nunique()
        ),
        "data_status": (
            "Census features available for all 32 districts"
        ),
    },
    {
        "research_question": "RQ2",
        "analysis_focus": (
            "NCD burden, WASH access, insurance, and "
            "breast-cancer examination indicators"
        ),
        "analysis_unit": "District",
        "observed_district_count": int(
            df["district_name"].nunique()
        ),
        "data_status": (
            "Selected indicators available for all 32 districts"
        ),
    },
    {
        "research_question": "RQ3",
        "analysis_focus": (
            "Vaccination, child nutrition, maternal care, "
            "and newborn follow-up indicators"
        ),
        "analysis_unit": "District",
        "observed_district_count": int(
            df["district_name"].nunique()
        ),
        "data_status": (
            "All 32 district profiles are available; one "
            "vaccination estimate is source-suppressed and "
            "handled separately in the modelling dataset"
        ),
    },
    {
        "research_question": "RQ4",
        "analysis_focus": (
            "Human-readable planning recommendations using "
            "combined demographic and health indicators"
        ),
        "analysis_unit": "District",
        "observed_district_count": int(
            df["district_name"].nunique()
        ),
        "data_status": (
            "District profiles are available; the recommendation "
            "layer has not yet been implemented"
        ),
    },
])

DISTRICT_COVERAGE_FILE = (
    TABLE_DIR
    / "table_district_coverage_by_research_question.csv"
)

district_coverage_by_rq.to_csv(
    DISTRICT_COVERAGE_FILE,
    index=False,
)

print("\nSaved district coverage table to:")
print(DISTRICT_COVERAGE_FILE)

def top_district_names(ranking_df, count=5):
    return ", ".join(
        ranking_df["district_name"]
        .head(count)
        .tolist()
    )


eda_insight_summary = pd.DataFrame([
    {
        "output": "Top 10 child-health burden districts",
        "key_observation": (
            f"{top_district_names(top_child_burden)} have the "
            "highest observed child-health burden scores."
        ),
        "planning_interpretation": (
            "These districts may warrant closer review for child "
            "nutrition, anaemia control, maternal-child health, "
            "and preventive outreach planning."
        ),
        "linked_research_question": "RQ3",
    },
    {
        "output": "Top 10 NCD burden districts",
        "key_observation": (
            f"{top_district_names(top_ncd_burden)} have the "
            "highest observed NCD burden scores."
        ),
        "planning_interpretation": (
            "These districts may warrant closer review of chronic-care "
            "capacity, NCD screening, follow-up systems, and diagnostic "
            "support."
        ),
        "linked_research_question": "RQ2",
    },
    {
        "output": "Top 10 WASH access-gap districts",
        "key_observation": (
            f"{top_district_names(top_wash_gap)} have the "
            "highest observed WASH access-gap scores."
        ),
        "planning_interpretation": (
            "These districts may warrant integrated review of "
            "sanitation, drinking water, clean fuel, PHC outreach, "
            "and preventive-care needs."
        ),
        "linked_research_question": "RQ2, RQ3, RQ4",
    },
    {
        "output": "Top 10 maternal-care gap districts",
        "key_observation": (
            f"{top_district_names(top_maternity_care_gap)} have the "
            "highest observed maternal-care gap scores."
        ),
        "planning_interpretation": (
            "These districts may warrant review of antenatal, skilled "
            "delivery, and early postnatal-care continuity."
        ),
        "linked_research_question": "RQ3, RQ4",
    },
    {
        "output": "Top 10 newborn follow-up gap districts",
        "key_observation": (
            f"{top_district_names(top_newborn_followup_gap)} have the "
            "highest observed newborn follow-up gap scores."
        ),
        "planning_interpretation": (
            "These districts may warrant review of early newborn "
            "follow-up and ANM or PHC outreach coverage."
        ),
        "linked_research_question": "RQ3, RQ4",
    },
    {
        "output": "Top 10 breast-cancer screening-gap districts",
        "key_observation": (
            "The mean breast-cancer screening gap is "
            f"{df['breast_cancer_screening_gap_percent'].mean():.1f}% "
            f"across {df['district_name'].nunique()} districts."
        ),
        "planning_interpretation": (
            "Low reported examination uptake may warrant review of "
            "awareness, outreach, and district-level diagnostic access."
        ),
        "linked_research_question": "RQ2, RQ4",
    },
    {
        "output": "Top 10 vaccination-gap districts",
        "key_observation": (
            f"{top_district_names(top_vaccination_gap)} have the "
            "highest observed vaccination gaps among the "
            f"{int(df['vaccination_gap_percent'].notna().sum())} "
            "available source estimates."
        ),
        "planning_interpretation": (
            "These districts may warrant review of immunization "
            "outreach and service-delivery coverage. The suppressed "
            "source estimate remains missing in the EDA dataset and "
            "is handled separately in the modelling dataset."
        ),
        "linked_research_question": "RQ3",
    },
    {
        "output": "Integrated EDA summary statistics",
        "key_observation": (
            f"The EDA dataset contains "
            f"{df['district_name'].nunique()} districts and combines "
            "demographic, access, child-health, maternal, newborn, "
            "NCD, screening, and vaccination indicators."
        ),
        "planning_interpretation": (
            "The integrated master supports district-level descriptive "
            "analysis; the separately validated modelling dataset is "
            "used for clustering."
        ),
        "linked_research_question": "RQ1, RQ2, RQ3, RQ4",
    },
])

EDA_INSIGHT_SUMMARY_FILE = (TABLE_DIR / "table_integrated_eda_insight_summary.csv")

eda_insight_summary.to_csv(EDA_INSIGHT_SUMMARY_FILE, index=False,)

print("\nSaved integrated EDA insight summary to:")
print(EDA_INSIGHT_SUMMARY_FILE)

print("\nIntegrated EDA complete.")