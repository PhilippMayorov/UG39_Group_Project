# Consulting Report: Canadian Household Housing Burden Analysis

**Prepared by:** UG39  
**Course:** DS 3000B, Introduction to Machine Learning  
**Institution:** University of Western Ontario  
**Date:** April 2026  
**Client:** [Client Name]  

---

## Executive Summary

Housing burden is unevenly distributed across Canada in ways that aggregate statistics do not reveal. This analysis recommends deploying the XGBoost predictive model as a geographic scoring tool to identify the 9,105 dissemination areas (16.0% of the national total) where households spend more than 30% of disposable income on shelter, and to direct intervention resources accordingly.

Applied to Environics Analytics HouseholdSpend 2025 data covering 56,857 Canadian dissemination areas at the PRCDDA level, this project used K-Means clustering, PCA, UMAP, an Elastic-Net GLM, XGBoost, and SHAP analysis to investigate the relationship between household spending patterns and housing burden.

Three findings stand out. First, housing burden is a non-linear problem: no single spending variable carries meaningful linear signal (maximum Pearson correlation of 0.010), and XGBoost (R² = 0.638) explains roughly twice the variance of the best linear model (ElasticNet R² = 0.319). Second, the most predictive features are transportation and discretionary spending categories, not shelter variables. Four of the ten highest-importance SHAP features are transportation-related, and the top feature functions as a reverse proxy for household financial capacity. Third, extreme burden is concentrated within the typical-spending majority, not the high-spending 2.6% minority isolated by clustering. Targeting programs at high spenders would miss the households most at risk.

Four recommendations follow: deploy XGBoost for neighbourhood-level risk scoring; expand the affordability metric to incorporate transportation costs; use low discretionary spending as an early-warning indicator; and target interventions at the typical-spending majority rather than spending-level proxies.

---

## Background

### Project Scope

This project was undertaken as part of DS 3000B (Introduction to Machine Learning) at the University of Western Ontario. The scope of the engagement was to investigate Canadian household housing affordability at the granular dissemination area (PRCDDA) level, which is Canada's finest-grained census geographic unit and is approximately equivalent to a neighbourhood block. The central analytical objective was twofold: to diagnose the socioeconomic and spending characteristics associated with high housing burden, and to build predictive machine learning models capable of estimating housing burden from observable household spending patterns.

The core metric used throughout this analysis is the **Housing Burden Ratio (HBR)**, defined as:

> **HBR = Total Shelter Expenditure (HSSH001S) / Household Disposable Income (HSAGDISPIN)**

An HBR value greater than 0.30 is widely regarded as an indicator of unaffordable housing, and the distribution of this ratio across Canadian dissemination areas forms the primary subject of this report.

### Commissioning Party

This analysis was commissioned as a graded deliverable for DS 3000B at the University of Western Ontario, requiring group UG39 to apply machine learning to proprietary data supplied by **Environics Analytics (EA)**. All data was provided under a Non-Disclosure Agreement and must be deleted upon course completion.

### Data Sources

All data used in this analysis was exclusively sourced from Environics Analytics. Four datasets were made available:

| Dataset | Description |
|---|---|
| **DemoStats 2025** | Demographic statistics per geographic area, including population, household structure, immigration, and income projections for 2020, 2025, 2028, 2030, and 2035 |
| **HouseholdSpend 2025** | Estimated household expenditure across hundreds of spending categories per PRCDDA, split across two files |
| **Geographic Hierarchy File 2025** | Lookup tables mapping geographic codes across the hierarchy from Province to CMA to CSD to PRCDDA |
| **Documentation** | Variable lists, metadata, and change logs for both DemoStats and HouseholdSpend datasets |

After filtering to the PRCDDA geographic level, the working dataset comprised approximately **44,000+ dissemination area observations**, each described by several hundred spending variables spanning food, transportation, recreation, health care, household operations, and other expenditure categories.

### Initial Assumptions and Data Exclusions

- **Target leakage prevention:** All shelter expenditure sub-items (`HSSH*` columns) were excluded from the predictor set, as these are components of the target variable (HBR) and their inclusion would constitute direct data leakage. Similarly, aggregate income and expenditure summary variables (`HSAGDISPIN`, `HSAGDISCIN`, `HSHNIAGG`, `HSTT001`, `HSTE001`, `HSTX001`, `HSTC001`) were removed.
- **Invalid observations:** Dissemination areas where household disposable income was reported as zero were excluded, as HBR is undefined in these cases. No missing values were found in the remaining dataset following geographic filtering.
- **Geographic scope:** Only PRCDDA-level rows were retained for analysis; higher-level geographic aggregates (provincial, CMA, CSD) were excluded to ensure observations were consistent and non-overlapping.
- **Feature set:** After applying all exclusions, approximately **316 numeric predictor features** were retained, drawn entirely from non-shelter household spending categories.

### Research Methods

The analytical pipeline was structured across three sequential phases:

1. **Data Preparation** covered geographic filtering, target variable construction, leakage removal, variable taxonomy development, and an 80/20 train/test split (random state 42). Features were standardised using a `StandardScaler` fitted exclusively on the training set.

2. **Unsupervised Learning** covered exploratory data analysis (EDA) including distributional analysis and correlation profiling; K-Means clustering (k = 2 to 12) evaluated via elbow method and silhouette scores; Principal Component Analysis (PCA) for dimensionality reduction and variance decomposition; and UMAP for non-linear manifold visualisation.

3. **Supervised Learning** covered an Elastic-Net Generalised Linear Model (GLM) with a Pearson Type III distributional transformation of the target variable evaluated via 5-fold cross-validated grid search; an XGBoost gradient-boosted tree regressor evaluated via randomised search; and SHAP (SHapley Additive exPlanations) analysis for model interpretability and feature attribution.

Model performance was assessed using R² on the held-out test set and bootstrapped 95% confidence intervals to account for sampling variability.

---

## Key Facts and Business Context

### Dataset Coverage and Burden Thresholds

The dataset covers **56,857 dissemination areas** across all Canadian provinces and territories, with a mean HBR of **0.2571** and a median of **0.2514**. The typical dissemination area therefore sits just below the standard 30% affordability threshold, but the distribution is heavily right-skewed (minimum 0.1169, maximum 4.4942), confirming a significant tail of severely burdened areas. Applying the threshold directly, **9,105 dissemination areas (16.0%)** have an HBR above 0.30 and **148 (0.3%)** have an HBR above 0.50, representing areas of severe burden where shelter consumes the majority of disposable income.

### Provincial Variation

Burden is not evenly distributed across Canada. British Columbia has the highest mean HBR (0.2895), with 34.5% of its dissemination areas above the 30% threshold. Ontario follows at a mean of 0.2738, with 22.6% of areas burdened; its large DA count of 20,305 means it contains the greatest absolute number of high-burden neighbourhoods nationally. Saskatchewan has a moderate mean HBR (0.2500) but the highest rate of severely burdened areas (0.9% above 0.50), indicating a concentrated tail risk. Quebec presents the sharpest contrast: despite being the second largest province by DA count (13,618), only 3.0% of its areas exceed the threshold, reflecting substantially more affordable neighbourhood-level conditions. These provincial differences are themselves averages that conceal within-province heterogeneity, which is the variation this analysis targets at the PRCDDA level.

---

## Findings

### 1. Data Overview

The merged dataset, filtered to the PRCDDA geographic level, contains 57,936 observations across 319 columns. After separating identifiers and the target variable and removing leakage columns, 316 numeric predictor features remained for modelling. The training set comprised 45,487 dissemination areas and the test set 11,370, reflecting an 80/20 random split. A missing value check confirmed zero missing entries across both sets, and a zero-variance check found no columns to remove, meaning every retained feature carries variation across dissemination areas.

The target variable, HBR, is heavily right-skewed with a skewness of 15.41. The distribution is tightly concentrated between 0.2 and 0.3, with a small number of extreme outliers extending past 4.0. This level of skew is significant because the majority of observations represent low-burden households, and models trained on this data may underperform on the relatively rare high-burden cases where intervention is most needed. The skew also directly motivated the distributional transformation applied in the ElasticNet model.

Examining individual correlations between features and HBR reveals that no single spending variable is a meaningful linear predictor. The highest absolute Pearson correlation across all 316 features is approximately 0.010. Furthermore, many of the top-correlated features are themselves highly correlated with each other; for example, HSHE032 and HSHE012C share a correlation of 0.98. This multicollinearity, combined with the weak individual signals, strongly suggests that the relationship between spending patterns and housing burden is complex and non-linear, motivating the use of both regularised linear models and tree-based methods.

---

### 2. Unsupervised Learning

#### 2a. K-Means Clustering

K-Means was run for k = 2 through 12. The elbow plot identified k = 5 as the algorithmic elbow, but the curve is smooth with no sharp bend, making the elbow method ambiguous here. Silhouette scores tell a clearer story: k = 2 produced the highest score of 0.8137, dropping monotonically to 0.1236 at k = 11. Since the silhouette score measures how well-separated clusters are relative to their internal cohesion, k = 2 was selected as the most defensible choice. The two methods do not fully agree, but the very high silhouette at k = 2 indicates a strong natural binary structure in the data.

With k = 2, the resulting clusters are highly imbalanced. **Cluster 0** contains 44,303 observations (approximately 97.4%) and **Cluster 1** contains 1,184 (approximately 2.6%). Comparing feature means across the two clusters reveals a consistent pattern: Cluster 1 shows spending values roughly 5 to 7 times higher than Cluster 0 across all examined categories. For example, HSTA009 averages 965,109 in Cluster 1 compared to 141,695 in Cluster 0. This pattern holds across recreation, home equipment, and household categories. Cluster 1 is therefore characterised as **"High-Spending Households"**, a small minority likely representing upper-income households, while Cluster 0 is characterised as **"Typical Households"** representing the broad majority of Canadian dissemination areas.

A boxplot of HBR by cluster shows similar median values across both groups, with nearly all extreme outliers (HBR > 1.0) falling in Cluster 0. This confirms that the clustering is driven by spending scale in the feature space rather than directly by housing burden. The two clusters capture the economic profile of households, not how burdened they are by shelter costs.

#### 2b. Principal Component Analysis

PCA was applied to the standardised 316-feature training matrix. The first principal component (PC1) alone captures 74.24% of total variance. By PC2 this rises to 79.41%, and the top 10 components cumulatively explain 93.64%. The steep jump at PC1 indicates that a single dominant direction accounts for most of the variation across all spending features.

The top five loadings on PC1 are from household operations (HSHO001), recreation (HSRE030), personal care (HSPC020, HSPC001), and communications (HSCS030), all with similar magnitudes of approximately 0.064. This uniformity identifies PC1 as a **general total spending magnitude axis** that increases as households spend more across all categories simultaneously. PC2 is driven by transport and clothing variables (HSTR051, HSCL014, HSHE012C), with loadings of 0.15 to 0.18, reflecting variation in vehicle and transit costs. PC3 follows a similar transportation theme covering HSRV005M1, HSTR009, and HSTR015.

The PC1 versus PC2 scatter plot, coloured by K-Means labels, shows the two clusters are clearly separated along PC1 only. Cluster 1 sits at a mean PC1 score of 66.65 compared to -1.78 for Cluster 0, a large gap confirming that the high-spending minority differs from the majority primarily in overall spending scale rather than spending composition.

#### 2c. UMAP

A parameter search across n_neighbors values of 5, 15, and 50 found that n_neighbors = 5 produced the highest 2D silhouette score of 0.1288, compared to 0.0381 and -0.0550 at the larger values. n_neighbors = 5 was therefore used for the final embedding.

The UMAP plot shows Cluster 0 forming dense, curved manifold structures across the 2D embedding, while Cluster 1 appears as scattered points distributed throughout rather than forming a compact region. This reflects how UMAP preserves local neighbourhood structure, as the small minority cluster does not project into a single tight area. The high-dimensional silhouette score (0.8137) is the more reliable measure of clustering quality here, as UMAP's non-linear compression introduces distortion when projecting a small, dispersed minority cluster into two dimensions.

---

### 3. Supervised Learning

#### 3a. Elastic-Net GLM

Because HBR is right-skewed and bounded, the target was transformed prior to modelling. A Pearson Type III (PT3) distribution was fitted to the training set using the distfit package (fitted parameters: skew = 15.41, loc = 0.257, scale = 0.048). Each training and test value was mapped through the PT3 cumulative distribution function and then converted to a standard normal quantile (z_Normal), producing an approximately normal regression target. Predictions were back-transformed through the same pipeline to recover values in the original HBR space.

GridSearchCV with 5-fold cross-validation was run over five alpha values in [10⁻⁴, 1] and five l1_ratio values in [0, 1]. The best configuration was **pure Ridge regression (l1_ratio = 0.0)**, meaning no sparsity was introduced and all 316 features received non-zero coefficients. This outcome is consistent with the multicollinearity observed in EDA, where Ridge spreads weight across correlated features rather than zeroing any out.

On the test set, the ElasticNet achieved an **R² of 0.3187** with a bootstrapped 95% confidence interval of [0.1772, 0.5476]. The largest coefficients include furnishing expenditure (negative, indicating homeowner financial stability) and household operations (positive), though opposing signs on correlated restaurant-spending variables indicate a Ridge multicollinearity artefact rather than a true directional effect.

#### 3b. XGBoost

XGBoost was trained directly on the original HBR target without distributional transformation, as tree-based models do not require normality assumptions. RandomizedSearchCV with 5-fold cross-validation and 30 iterations searched over n_estimators (200 to 800), max_depth (2 to 6), learning_rate (0.01 to 0.1), subsample (0.6 to 1.0), and colsample_bytree (0.6 to 1.0). This search space was designed to balance model complexity, regularisation through subsampling, and runtime.

On the test set, XGBoost achieved an **R² of 0.6380**, MSE of 0.001582, and RMSE of 0.039768, with a bootstrapped 95% confidence interval of [0.4956, 0.8764]. Compared to the ElasticNet, this represents an improvement of **+0.3193 in R²**. The confidence intervals overlap only narrowly (between 0.4956 and 0.5476), with most of the XGBoost interval falling above ElasticNet's range, indicating meaningfully better predictive performance. The scatterplot shows XGBoost tracks the dense low-to-mid burden range more accurately, though it still underpredicts the rare extreme high-burden observations.

XGBoost is the recommended model for prediction. ElasticNet remains a useful interpretable baseline, but the near-doubling of explained variance demonstrates that linear modelling underfits this problem.

#### 3c. SHAP Explainability

SHAP TreeExplainer was applied to the best XGBoost model on the test set. The top 10 features by mean absolute SHAP value are:

| Rank | Feature | Description | Mean |SHAP| |
|---|---|---|---|
| 1 | HSRV003B | Purchase of snowmobiles | 0.0191 |
| 2 | HSHO002 | Domestic and custodial services | 0.0079 |
| 3 | HSHE012C | Parts and accessories for garden tools | 0.0066 |
| 4 | HSMG010 | Alimony and child support | 0.0054 |
| 5 | HSHC063 | Cannabis for medical use | 0.0044 |
| 6 | HSTR013 | Regular fees for leased vans | 0.0037 |
| 7 | HSTR055 | Inter-city bus | 0.0037 |
| 8 | HSCS007 | Internet access services | 0.0036 |
| 9 | HSTR051 | City/commuter bus, subway, streetcar | 0.0032 |
| 10 | HSTR038 | Parking costs | 0.0030 |

Transportation features account for 4 of the top 10, confirming that mobility-related costs are the strongest category of predictors for housing burden. The top feature, HSRV003B (snowmobile purchases), functions as a financial-capacity proxy. Households with low values on this variable, indicating they cannot afford discretionary recreational spending, are consistently associated with higher predicted HBR, while households with high values show reduced predicted burden. A similar pattern holds for HSHO002 and HSHE012C, where lower discretionary spending acts as a reverse proxy for household financial health.

HSMG010 (alimony and child support) captures a distinct mechanism: fixed legal obligations reduce the disposable budget available for housing. Transit features (HSTR051, HSTR055, HSTR038) show the opposite direction, where higher commuting costs are associated with higher predicted burden, consistent with households in high-cost areas relying on public transport.

Critically, there is **zero overlap** between the top 10 SHAP features and the top 10 ElasticNet coefficients. This divergence, combined with the R² gap of +0.3193, provides strong evidence that housing burden prediction is a substantially non-linear problem. The two models are identifying entirely different predictive structures in the data, and the non-linear model accounts for roughly twice the variance.

---

## Recommendations

### R1. Deploy XGBoost as a Geographic Risk-Scoring Tool

To help policymakers identify which dissemination areas face the greatest housing affordability pressure, we recommend deploying the XGBoost model as an operational geographic scoring tool applied to the full national PRCDDA dataset. At an R² of 0.6380 on the held-out test set, the model explains roughly two-thirds of the variance in HBR across Canadian dissemination areas and can produce a ranked list of neighbourhoods by predicted burden. This enables housing support programs and municipal planning bodies to allocate intervention resources toward the highest-risk areas rather than relying on aggregate provincial or city-level statistics, which conceal the neighbourhood-level variation that this analysis has demonstrated is substantial.

### R2. Expand the Affordability Definition to Include Transportation Costs

To help produce a more accurate picture of total household cost burden, we recommend that future affordability assessments incorporate transportation expenditure alongside shelter costs rather than treating them independently. Four of the ten most predictive SHAP features are transportation-related: regular fees for leased vans (HSTR013), inter-city bus spending (HSTR055), city transit spending (HSTR051), and parking costs (HSTR038). Higher spending across these variables is consistently associated with higher predicted HBR, which is consistent with households in high-cost urban areas relying heavily on paid transit and incurring both high shelter and high mobility costs simultaneously. A combined shelter-plus-transport burden ratio would better reflect the true financial pressure on these households and would be more actionable for transit and land-use policy.

### R3. Use Low Discretionary Spending as an Early-Warning Indicator

To help identify financially stressed households before they appear in arrears or social support data, we recommend treating low discretionary spending as a proxy screening signal for elevated housing burden risk. The SHAP analysis shows that households with near-zero expenditure on snowmobile purchases (HSRV003B), domestic and custodial services (HSHO002), and garden equipment accessories (HSHE012C) are consistently predicted to have higher HBR values, with HSRV003B carrying the single largest mean absolute SHAP value (0.0191) in the model. These are items that financially comfortable households purchase routinely; their absence signals constrained budgets. Applied to current Environics Analytics data, a simple threshold-based flag on this cluster of variables could serve as a low-cost early-warning screen for outreach programs.

### R4. Target Housing Interventions at the Broad Majority, Not the High-Spending Minority

To help ensure that affordability programs reach the households most in need, we recommend against using overall spending level as the primary criterion for program eligibility or prioritisation. The K-Means analysis produced a clear binary split: a high-spending minority (Cluster 1, approximately 2.6% of dissemination areas) with expenditure five to seven times that of the majority, and a broad typical majority (Cluster 0, approximately 97.4%). Critically, nearly all extreme HBR outliers (HBR above 1.0) fall within Cluster 0, not within the high-spending group. Housing burden is concentrated in the ordinary-spending majority, and any screening approach that uses high spending as a signal of vulnerability will systematically miss the households that are most at risk.

---