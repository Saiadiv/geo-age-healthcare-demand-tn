# Geo-Age Healthcare Demand Forecasting and Resource Allocation Decision-Support System for Tamil Nadu

**Project Type:** QM640 Data Analytics Capstone  
**Author:** Sai Adithya Venkatesulu  
**Institution:** Walsh College  

## Project Overview

This project develops a healthcare demand forecasting and resource allocation decision-support system for Tamil Nadu. The goal is to integrate trusted public datasets such as Census India, NFHS-5, HMIS, health facility records, infrastructure reports, and population projection sources to identify underserved healthcare demand clusters.

The core of the project is **data harmonization**. The system brings fragmented public datasets into one unified analytical database. This unified database will support district-level healthcare demand analysis by combining demographic structure, public-health indicators, service-utilization patterns, facility availability, and future population demand signals.

Machine learning will then be used for **geographic clustering, priority scoring, and demand forecasting**. The clustering layer will group districts with similar healthcare needs, the priority-scoring layer will identify areas that may require urgent review, and the forecasting layer will estimate future demand trends where projection or time-series data is available.

An AI-assisted recommendation layer will convert analytical findings into human-readable suggestions for healthcare planners. These recommendations may support decisions related to doctor deployment, mobile clinics, vaccination drives, diabetes and hypertension screening camps, telemedicine support, and facility strengthening.

## Decision-Support Statement

This project is a **decision-support system**, not an autonomous clinical or administrative decision system. It does not diagnose patients, prescribe treatment, or automatically allocate healthcare resources. All recommendations require human review by authorized healthcare planners, doctors, district health officials, or public-health administrators.

## Core Data Sources

1. Census India C-13 and C-14 age-specific population data
2. National Family Health Survey NFHS-5
3. Health Management Information System HMIS
4. Health Facility Registry under Ayushman Bharat Digital Mission
5. Health Dynamics of India infrastructure and human resources data
6. State or central government health department publications
7. Population projection sources, if required

## Core Framework

```text
Fragmented trusted public data sources
        ↓
Data cleaning and district harmonization
        ↓
Unified district-level analytical database
        ↓
Feature engineering
        ↓
Machine learning for clustering, priority scoring, and forecasting
        ↓
Dashboard and geospatial visualization
        ↓
AI-assisted recommendation layer
        ↓
Human decision-maker review
ls
