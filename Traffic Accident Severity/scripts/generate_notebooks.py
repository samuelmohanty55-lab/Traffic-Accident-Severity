"""
Generates clean, production-ready Jupyter Notebooks for Phases 1 to 9.
"""

import nbformat as nbf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 3: Exploratory Data Analysis & Data Understanding
## Project: Road Traffic Accident Severity Prediction & Risk Factor Identification

This notebook conducts comprehensive exploratory data analysis on the **US Accidents** dataset to uncover:
1. Missing value patterns and schema distributions
2. Target distribution and severity class imbalance
3. Temporal trends (Hour of day, rush hours, weekend patterns)
4. Environmental & weather risk conditions (Visibility, precipitation, temperature)
5. Road infrastructure correlations (Junctions, traffic signals, crossings)
"""
        ),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DATA_DIR, TARGET_COLUMN
from src.data_loader import load_dataset

# Plot formatting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
"""
        ),
        nbf.v4.new_markdown_cell("### 1. Load Dataset"),
        nbf.v4.new_code_cell(
            """df = load_dataset(sample_size=30000)
print("Dataset Shape:", df.shape)
df.head()
"""
        ),
        nbf.v4.new_markdown_cell("### 2. Schema and Missing Values Inspection"),
        nbf.v4.new_code_cell(
            """missing_stats = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Percentage (%)': (df.isnull().sum() / len(df)) * 100
}).sort_values(by='Missing Percentage (%)', ascending=False)

missing_stats.head(15)
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Severity Distribution & Class Imbalance Analysis\nNotice the extreme rarity of severe/fatal accidents."),
        nbf.v4.new_code_cell(
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Original 4-class severity
df['Severity'].value_counts(normalize=True).sort_index().plot(
    kind='bar', color='#3b82f6', edgecolor='black', ax=ax1
)
ax1.set_title("Original Severity Distribution (Classes 1-4)")
ax1.set_xlabel("Severity Level")
ax1.set_ylabel("Proportion")

# Binary target (Serious vs Non-Serious)
is_serious = (df['Severity'] >= 3).astype(int)
is_serious.value_counts(normalize=True).rename({0: 'Non-Serious (1-2)', 1: 'Serious (3-4)'}).plot(
    kind='bar', color=['#10b981', '#ef4444'], edgecolor='black', ax=ax2
)
ax2.set_title("Binary Target Distribution")
ax2.set_ylabel("Proportion")

plt.tight_layout()
plt.show()
print(f"Serious Class Proportion: {is_serious.mean()*100:.2f}%")
"""
        ),
        nbf.v4.new_markdown_cell("### 4. Severity vs Temporal Drivers (Hour, Rush Hour, Day of Week)"),
        nbf.v4.new_code_cell(
            """df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.day_name()
df['Is_Serious'] = (df['Severity'] >= 3).astype(int)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Hourly risk
hourly_risk = df.groupby('Hour')['Is_Serious'].mean() * 100
hourly_risk.plot(kind='line', marker='o', color='#dc2626', lw=2.5, ax=ax1)
ax1.set_title("Serious Accident Rate by Hour of Day (%)")
ax1.set_xlabel("Hour (0-23)")
ax1.set_ylabel("Serious Rate (%)")
ax1.grid(True)

# Day of week risk
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_risk = df.groupby('DayOfWeek')['Is_Serious'].mean().reindex(order) * 100
day_risk.plot(kind='bar', color='#4f46e5', edgecolor='black', ax=ax2)
ax2.set_title("Serious Accident Rate by Day of Week (%)")
ax2.set_ylabel("Serious Rate (%)")

plt.tight_layout()
plt.show()
"""
        ),
        nbf.v4.new_markdown_cell("### 5. Severity vs Weather & Visibility"),
        nbf.v4.new_code_cell(
            """plt.figure(figsize=(10, 5))
sns.boxplot(x='Severity', y='Visibility(mi)', data=df, palette='Blues')
plt.title("Visibility (miles) Distribution across Severity Levels")
plt.ylim(0, 11)
plt.show()
"""
        ),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "01_eda_and_data_understanding.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def create_preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 4 & 5: Data Preprocessing & Feature Engineering
## Scikit-Learn Pipeline & ColumnTransformer Architecture

This notebook demonstrates the end-to-end preprocessing pipeline:
1. Feature extraction from timestamps (Hour, Day of Week, Rush Hour, Night/Day, Season)
2. Weather aggregation & physical hazard flags (Freezing, Low Visibility, High Wind)
3. Purging leakage columns and uninformative metadata
4. Stratified Train / Validation / Test split (70 / 15 / 15) executed strictly before scaling or resampling
5. Building a production `ColumnTransformer` with median imputation and one-hot encoding
"""
        ),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
from src.data_loader import load_dataset
from src.preprocessor import prepare_raw_data, get_feature_lists, build_preprocessor_pipeline, stratified_split

# Load raw data
raw_df = load_dataset(sample_size=30000)
print("Raw input shape:", raw_df.shape)
"""
        ),
        nbf.v4.new_markdown_cell("### 1. Apply Feature Engineering and Leakage Purging"),
        nbf.v4.new_code_cell(
            """clean_df = prepare_raw_data(raw_df)
print("Cleaned DataFrame shape:", clean_df.shape)
clean_df.head(3)
"""
        ),
        nbf.v4.new_markdown_cell("### 2. Stratified Data Split (70% Train, 15% Val, 15% Test)"),
        nbf.v4.new_code_cell(
            """X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(clean_df)
print(f"Train samples: {len(X_train)} | Serious rate: {y_train.mean():.3f}")
print(f"Val samples:   {len(X_val)} | Serious rate: {y_val.mean():.3f}")
print(f"Test samples:  {len(X_test)} | Serious rate: {y_test.mean():.3f}")
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Build & Fit ColumnTransformer on Training Data"),
        nbf.v4.new_code_cell(
            """num_cols, cat_cols, road_cols = get_feature_lists(X_train)
preprocessor = build_preprocessor_pipeline(num_cols, cat_cols, road_cols, scale_numeric=True)

# Fit strictly on train
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

print("Processed feature matrix shape:", X_train_proc.shape)
"""
        ),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "02_preprocessing_and_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def create_training_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 6 & 7: Model Training, Imbalance Handling & Benchmarking
## Comparing Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and MLP

Key Objectives:
1. Benchmark 6 diverse model families
2. Implement class imbalance mitigation (Balanced class weighting & threshold tuning)
3. Evaluate using **Macro F1, Minority Serious Recall, and PR-AUC** (not accuracy alone)
4. Plot side-by-side ROC curves, Precision-Recall curves, and confusion matrices
"""
        ),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
from src.config import REPORTS_DIR, FIGURES_DIR
from src.models import train_and_evaluate_all_models
from src.data_loader import load_dataset
from src.preprocessor import prepare_raw_data, get_feature_lists, build_preprocessor_pipeline, stratified_split

# Prepare data
df = prepare_raw_data(load_dataset(sample_size=30000))
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)
num_cols, cat_cols, road_cols = get_feature_lists(X_train)
preprocessor = build_preprocessor_pipeline(num_cols, cat_cols, road_cols, scale_numeric=True)

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
"""
        ),
        nbf.v4.new_markdown_cell("### 1. Train and Benchmark All Models"),
        nbf.v4.new_code_cell(
            """results_df, fitted_models, best_thresholds = train_and_evaluate_all_models(
    X_train_proc, y_train.values, X_val_proc, y_val.values, imbalance_strategy="class_weight"
)
results_df
"""
        ),
        nbf.v4.new_markdown_cell("### 2. View Performance Comparison Table"),
        nbf.v4.new_code_cell(
            """comp_csv = REPORTS_DIR / "model_comparison.csv"
if comp_csv.exists():
    display_df = pd.read_csv(comp_csv)
    display(display_df.style.highlight_max(subset=['Macro F1', 'Recall (Serious)', 'ROC-AUC', 'PR-AUC'], color='#bfdbfe'))
"""
        ),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "03_model_training_and_comparison.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def create_shap_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            """# Phase 8: Risk Factor Identification & Explainable AI (SHAP)
## Uncovering Conditions Linked to Serious Road Traffic Accidents

This notebook addresses the second half of the project scope:
1. Computing SHAP (SHapley Additive exPlanations) values to quantify feature impact directionality
2. Generating Beeswarm Summary and Global Feature Importance plots
3. Extracting plain-English decision rules from a shallow tree
4. Computing statistical Odds Ratios (OR) and 95% Confidence Intervals for hazard conditions
"""
        ),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

import pandas as pd
from src.config import REPORTS_DIR, FIGURES_DIR
from src.explainability import calculate_risk_odds_ratios
from src.data_loader import load_dataset
from src.preprocessor import prepare_raw_data

df = prepare_raw_data(load_dataset(sample_size=30000))
"""
        ),
        nbf.v4.new_markdown_cell("### 1. Statistical Odds Ratios (Relative Risk Analysis)"),
        nbf.v4.new_code_cell(
            """odds_df = calculate_risk_odds_ratios(df)
odds_df[['Risk Factor', 'Odds Ratio (OR)', '95% CI Lower', '95% CI Upper', 'Rate Exposed (%)', 'Rate Baseline (%)']]
"""
        ),
        nbf.v4.new_markdown_cell("### 2. Plain-Language Practical Takeaways"),
        nbf.v4.new_code_cell(
            """for statement in odds_df['Plain-Language Finding']:
    print("•", statement)
"""
        ),
        nbf.v4.new_markdown_cell("### 3. Extracted Decision Rules"),
        nbf.v4.new_code_cell(
            """rules_file = REPORTS_DIR / "decision_rules.txt"
if rules_file.exists():
    with open(rules_file, 'r') as f:
        print(f.read())
"""
        ),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "04_risk_factor_identification_shap.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_eda_notebook()
    create_preprocessing_notebook()
    create_training_notebook()
    create_shap_notebook()
    print("[Notebooks] Successfully generated all 4 Jupyter Notebooks in notebooks/")
