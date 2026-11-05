# 🚗 Road Traffic Accident Severity Prediction & Risk Factor Identification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit-red.svg)](https://streamlit.io/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP-green.svg)](https://shap.readthedocs.io/)

An end-to-end Machine Learning and Explainable AI (XAI) system designed to **predict traffic accident severity** and **systematically identify high-risk environmental, temporal, and road geometry factors** linked to serious accidents.

---

## 📌 1. Project Scoping & Core Objectives

### The Problem
Traditional accident analysis often suffers from two fundamental flaws:
1. **Severe Class Imbalance:** Serious and fatal accidents represent a small fraction (~15–23%) of all incidents. A naive model predicting "non-serious" every time achieves ~80% accuracy while failing 100% of emergency dispatch cases.
2. **Black-Box Predictions:** Knowing an accident *will* be serious without knowing *why* does not help transportation departments design safer roads or deploy proactive patrols.

### Our Solution
- **Target Formulation:** Binary classification (`Is_Serious`: 1 for Severity 3 & 4, 0 for Severity 1 & 2) focused on incidents causing major road blockages and severe injury.
- **Evaluation Priority:** Optimized for **Macro F1, Serious Class Recall, and PR-AUC** (Average Precision) rather than raw accuracy.
- **Explainability (Phase 8):** Combined **SHAP (SHapley Additive exPlanations)** with **Epidemiological Odds Ratios (OR)** to produce plain-language risk statements (e.g., *"Low visibility (<2 mi) increases odds of a serious accident by 4.02x"*).

---

## 📂 2. Repository Architecture

```
ML/
├── app/
│   └── app.py                     # Interactive Streamlit Web App (Predictor & XAI Dashboard)
├── data/
│   ├── raw/                       # Raw input data (e.g. US_Accidents_Sample.csv)
│   └── processed/                 # Cached splits and feature matrices
├── models/
│   ├── best_model_pipeline.joblib # Serialized production inference pipeline
│   └── model_metadata.json        # Hyperparameters, optimal thresholds & test scores
├── notebooks/
│   ├── 01_eda_and_data_understanding.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_comparison.ipynb
│   └── 04_risk_factor_identification_shap.ipynb
├── reports/
│   ├── figures/                   # High-res publication plots (ROC, PR, SHAP beeswarm)
│   ├── model_comparison.csv       # Benchmark results across all 6 algorithms
│   ├── final_test_evaluation.csv  # Final score on held-out test partition
│   ├── risk_factor_odds_ratios.csv# Statistical Odds Ratios and Confidence Intervals
│   ├── risk_factors_findings.md   # Plain-language risk factor conclusions
│   └── decision_rules.txt         # Extracted shallow decision tree rules
├── src/
│   ├── __init__.py
│   ├── config.py                  # Central configuration and feature schemas
│   ├── data_loader.py             # Data loading and representative sample generator
│   ├── feature_engineering.py     # Temporal, weather, and road feature engineering
│   ├── preprocessor.py            # Leakage-free ColumnTransformer pipeline
│   ├── models.py                  # Model instantiations, SMOTE, class weighting
│   ├── evaluation.py              # Metrics, threshold tuning, and ROC/PR plotting
│   ├── explainability.py          # SHAP beeswarm/bar plots and Odds Ratio engine
│   └── train.py                   # Master pipeline orchestrator
├── requirements.txt               # Pinned Python package dependencies
├── project_roadmap_schedule.md    # Week-by-week project execution timeline
└── README.md
```

---

## 🚀 3. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone <your-repo-url>
cd ML

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the End-to-End Pipeline
```bash
python -m src.train
```
*Note: If no raw CSV is placed in `data/raw/`, the data loader automatically synthesizes a statistically faithful 30,000-record dataset matching the US Accidents schema so the entire pipeline runs immediately!*

### Step 3: Launch the Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser to access:
- **Real-Time Severity Predictor:** Sliders and toggles for weather, road features, and time with dynamic risk calculation.
- **Model Benchmarks:** Comparison tables, ROC curves, and confusion matrices.
- **SHAP & Risk Factors:** Beeswarm plots, feature importance, and epidemiological odds ratios.

---

## 📊 4. Model Comparison & Benchmark Results

Evaluated on stratified validation data with class-weight balancing and validation threshold tuning:

| Model Architecture | Default F1 | Tuned Threshold | Serious Recall | Serious Precision | Macro F1 | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** *(Champion)* | 0.410 | **0.510** | **59.2%** | **31.5%** | **0.559** | **0.338** | **0.629** |
| **Random Forest** | 0.397 | 0.450 | 62.4% | 30.5% | 0.544 | 0.338 | 0.626 |
| **Decision Tree** | 0.407 | 0.510 | 60.2% | 30.9% | 0.552 | 0.320 | 0.618 |
| **LightGBM** | 0.392 | 0.440 | 64.1% | 29.3% | 0.528 | 0.336 | 0.617 |
| **XGBoost** | 0.389 | 0.430 | 69.2% | 28.8% | 0.511 | 0.336 | 0.622 |
| **MLP Neural Net** | 0.006 | 0.180 | 75.8% | 27.1% | 0.465 | 0.308 | 0.606 |

### 🔑 Critical Insight: Why Threshold Tuning Matters
At standard default threshold `0.50`, the **MLP Neural Network** achieved **76.9% accuracy**, but its recall on serious accidents was **0.28%** (virtually useless).
By tuning the probability decision threshold to **0.18**, minority class recall skyrocketed from **0.28% to 75.8%**, proving why raw accuracy is misleading on imbalanced datasets.

---

## 🔍 5. Phase 8: Key Risk Factor Findings

Using SHAP explainability and Epidemiological Odds Ratios ($OR = \frac{ad}{bc}$):

| Hazard Condition | Odds Ratio (OR) | 95% Confidence Interval | Serious Accident Rate | Baseline Rate | Plain-Language Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Low Visibility (< 2 mi)** | **4.02x** | [3.47, 4.65] | **53.5%** | 22.3% | **4.02x higher odds** of severe outcome |
| **Nighttime (8 PM – 6 AM)** | **1.90x** | [1.80, 2.00] | **29.8%** | 18.3% | Nearly double the odds of serious severity |
| **Highway Junction** | **1.90x** | [1.77, 2.03] | **33.9%** | 21.3% | Interchanges create high-speed merge risks |
| **Rush Hour Commute** | **0.69x** | [0.64, 0.74] | **18.0%** | 24.2% | Lower severity due to congested, low-speed traffic |
| **Traffic Signal Present**| **0.66x** | [0.61, 0.71] | **17.4%** | 24.3% | Regulated intersections reduce fatal collisions |

---

## 🛡️ 6. Prevention of Data Leakage (Viva Defense Highlight)

1. **Strict Feature Purging:** Columns known only *after* the collision (casualty counts, injury codes, incident end duration) are eliminated during ingestion.
2. **Split-Before-Transform:** The 70/15/15 stratified train/val/test split occurs before any imputation, scaling, or oversampling.
3. **Pipeline Encapsulation:** Preprocessing parameters (median values, standard scaler mean/variance, one-hot categories) are strictly learned from the training fold and applied transitively to validation and test data.

---

## 📜 7. License & Citation
Developed for academic research and road safety analytics. Open sourced under the [MIT License](LICENSE).
