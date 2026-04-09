# Consulting Report: Canadian Household Housing Burden Analysis

**Prepared by:** UG39  
**Course:** DS 3000B — Introduction to Machine Learning  
**Institution:** University of Western Ontario  
**Date:** April 2026  
**Client:** [Client Name]  

---

## Executive Summary

*[To be written last — summarise the key findings and recommendations using the BLUF method.]*

---

## Background

### Project Scope

This project was undertaken as part of DS 3000B (Introduction to Machine Learning) at the University of Western Ontario. The scope of the engagement was to investigate Canadian household housing affordability at the granular dissemination area (PRCDDA) level — Canada's finest-grained census geographic unit, approximately equivalent to a neighbourhood block. The central analytical objective was twofold: to diagnose the socioeconomic and spending characteristics associated with high housing burden, and to build predictive machine learning models capable of estimating housing burden from observable household spending patterns.

The core metric used throughout this analysis is the **Housing Burden Ratio (HBR)**, defined as:

> **HBR = Total Shelter Expenditure (HSSH001S) / Household Disposable Income (HSAGDISPIN)**

An HBR value greater than 0.30 is widely regarded as an indicator of unaffordable housing, and the distribution of this ratio across Canadian dissemination areas forms the primary subject of this report.

### Commissioning Party

This analysis was commissioned as a graded coursework deliverable for DS 3000B. The assignment brief was issued by the course instructors and required the student group UG39 to apply machine learning techniques — spanning data preparation, unsupervised learning, and supervised learning — to proprietary data supplied by **Environics Analytics (EA)**, a leading Canadian data and analytics company. All data was provided under a Non-Disclosure Agreement (NDA) with the University of Western Ontario and is subject to legal restrictions; the data must be deleted upon course completion and may not be shared externally.

### Research Team

The analysis was conducted collaboratively by the members of **Group UG39**. Work was divided across modular feature branches in a shared Git repository, with each team member contributing to distinct stages of the analytical pipeline before integration into a unified deliverable.

### Data Sources

All data used in this analysis was exclusively sourced from Environics Analytics. Four datasets were made available:

| Dataset | Description |
|---|---|
| **DemoStats 2025** | Demographic statistics per geographic area, including population, household structure, immigration, and income projections for 2020, 2025, 2028, 2030, and 2035 |
| **HouseholdSpend 2025** | Estimated household expenditure across hundreds of spending categories per PRCDDA, split across two files |
| **Geographic Hierarchy File 2025** | Lookup tables mapping geographic codes across the hierarchy: Province → CMA → CSD → PRCDDA |
| **Documentation** | Variable lists, metadata, and change logs for both DemoStats and HouseholdSpend datasets |

After filtering to the PRCDDA geographic level, the working dataset comprised approximately **44,000+ dissemination area observations**, each described by several hundred spending variables spanning food, transportation, recreation, health care, household operations, and other expenditure categories.

### Initial Assumptions and Data Exclusions

Several key assumptions and exclusions were established prior to modelling to ensure methodological integrity:

- **Target leakage prevention:** All shelter expenditure sub-items (`HSSH*` columns) were excluded from the predictor set, as these are components of the target variable (HBR) and their inclusion would constitute direct data leakage. Similarly, aggregate income and expenditure summary variables (`HSAGDISPIN`, `HSAGDISCIN`, `HSHNIAGG`, `HSTT001`, `HSTE001`, `HSTX001`, `HSTC001`) were removed.
- **Invalid observations:** Dissemination areas where household disposable income was reported as zero were excluded, as HBR is undefined in these cases. No missing values were found in the remaining dataset following geographic filtering.
- **Geographic scope:** Only PRCDDA-level rows were retained for analysis; higher-level geographic aggregates (provincial, CMA, CSD) were excluded to ensure observations were consistent and non-overlapping.
- **Feature set:** After applying all exclusions, approximately **316 numeric predictor features** were retained, drawn entirely from non-shelter household spending categories.

### Research Methods

The analytical pipeline was structured across three sequential phases:

1. **Data Preparation** — Geographic filtering, target variable construction, leakage removal, variable taxonomy development, and an 80/20 train/test split (random state 42). Features were standardised using a `StandardScaler` fitted exclusively on the training set.

2. **Unsupervised Learning** — Exploratory data analysis (EDA) including distributional analysis and correlation profiling; K-Means clustering (k = 2 to 12) evaluated via elbow method and silhouette scores; Principal Component Analysis (PCA) for dimensionality reduction and variance decomposition; and UMAP for non-linear manifold visualisation.

3. **Supervised Learning** — An Elastic-Net Generalised Linear Model (GLM) with a Pearson Type III distributional transformation of the target variable, evaluated via 5-fold cross-validated grid search; an XGBoost gradient-boosted tree regressor evaluated via randomised search; and SHAP (SHapley Additive exPlanations) analysis for model interpretability and feature attribution.

Model performance was assessed using R² on the held-out test set and bootstrapped 95% confidence intervals to account for sampling variability.

---

## Key Facts and Business Context

*[To be completed — include standout quantitative statistics, the housing affordability landscape in Canada, and why this analysis is relevant to policymakers or stakeholders.]*

---

## Findings

*[To be completed — present findings from EDA, clustering, PCA, UMAP, ElasticNet, and XGBoost using the MECE structure. Include charts and model performance metrics.]*

---

## Recommendations

*[To be completed — provide concrete, actionable next steps for the commissioning party based on the findings. Follow the format: "To help with X, we recommend Y, because of Z."]*

---

## Appendices

*[To be completed — include supplementary tables, extended model outputs, variable lists, or any supporting material referenced in the main body.]*
