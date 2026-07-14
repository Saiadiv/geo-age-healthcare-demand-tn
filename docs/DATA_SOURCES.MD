# Data Sources

This document tracks the trusted public datasets planned for the Geo-Age Healthcare Demand Forecasting and Resource Allocation Decision-Support System for Tamil Nadu.

## Data Source Summary

| Dataset | Source | Geography | Year / Period | Key Variables | Use in Project | Limitation |
|---|---|---|---|---|---|---|
| Census C-13 | Census India | State and district | 2011 | Single-year age, sex, rural/urban residence | Detailed age-specific healthcare demand estimation | Older data; requires projection or careful interpretation |
| Census C-14 | Census India | State and district | 2011 | Five-year age groups, sex, rural/urban residence | Clean age-band analysis for child, adult, and elderly demand | Older data; district boundaries may differ from current structure |
| NFHS-5 India | IIPS / Ministry of Health and Family Welfare | National, state, district | 2019–21 | Health, nutrition, immunization, diabetes, hypertension, maternal health | Public-health burden and preventive care indicators | Survey-based; not real-time |
| NFHS-5 Tamil Nadu | IIPS / Ministry of Health and Family Welfare | Tamil Nadu districts | 2019–21 | Tamil Nadu district-level health indicators | District-level disease and preventive-care demand features | Based on older district structure |
| HMIS | Ministry of Health and Family Welfare | Facility, block, district, state | Varies by availability | Service-utilization indicators, immunization, maternal health, outpatient services | Compare estimated demand against actual service utilization | Reporting quality and completeness may vary |
| Health Facility Registry | Ayushman Bharat Digital Mission | Facility-level | Current / dynamic | Hospitals, clinics, labs, diagnostic centres, pharmacies | Healthcare supply and facility availability analysis | Data access/export may require additional processing |
| Health Dynamics of India | Ministry of Health and Family Welfare | State and district | 2022–23 | Sub-centres, PHCs, CHCs, district hospitals, medical colleges, health infrastructure | Facility and workforce capacity indicators | Facility count may not fully capture quality or real availability |
| Population Projections | Government / UN sources | State-level or broader | 2011–2036 or latest available | Projected population by age group | Future healthcare demand estimation | Projection data may not be available at detailed district level |

## Data Harmonization Plan

The first version of the project will use a 32-district harmonized Tamil Nadu structure because Census 2011 and NFHS-5 are more compatible with this structure.

A future extension will map the system to Tamil Nadu’s current 38-district administrative structure using a district crosswalk and updated facility/service datasets.

## Core Data Harmonization Tasks

1. Standardize district names across datasets.
2. Align district boundaries and reporting structures.
3. Convert age data into common age groups.
4. Create derived indicators such as child share, elderly share, rural share, and facility density.
5. Merge population, health, service-utilization, and facility data into a unified district-level analytical database.
6. Document missingness, uncertainty, and source limitations.

## Planned Unified Dataset

The final harmonized dataset will be stored as:

```text
data/processed/district_healthcare_demand_master.csv. 