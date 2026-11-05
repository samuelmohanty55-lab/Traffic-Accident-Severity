# 📅 Road Accident Severity Project: Week-by-Week Execution Schedule

A compressed, actionable roadmap aligned with your **Day 1 to 25 project lifecycle**, outlining exact deliverables, milestones, and viva presentation prep for each phase.

---

## 🗓️ Week 1: Foundation, Data Ingestion & EDA (Days 1–7)

### Objectives:
- Complete environment setup and repository scaffolding.
- Ingest and sample the primary dataset (US Accidents ~200k–500k rows).
- Perform deep Exploratory Data Analysis (EDA) and quantify class imbalance.

| Day Range | Phase | Key Tasks & Checklists | Deliverable |
| :--- | :--- | :--- | :--- |
| **Day 1–2** | **Phase 1: Setup & Scoping** | • Formulate target as Binary (`Is_Serious`: Severity 3–4 vs 1–2)<br>• Lock primary metrics: Macro F1, Serious Recall, PR-AUC<br>• Initialize Git repo & environment (`requirements.txt`) | Config file: `src/config.py`<br>Environment setup |
| **Day 2–3** | **Phase 2: Data Collection** | • Download Kaggle US Accidents CSV to `data/raw/`<br>• Use chunked streaming sampler to extract 200k–500k records<br>• Validate schema and null value distribution | Ingestion script: `src/data_loader.py`<br>`data/raw/US_Accidents_Sample.csv` |
| **Day 3–5** | **Phase 3: Data Understanding & EDA** | • Analyze missingness (flag columns with >40% missing)<br>• Plot hourly, day-of-week, and seasonal distributions<br>• Plot severity vs. visibility, precipitation, road surface<br>• Generate correlation matrix heatmap | Notebook: `01_eda_and_data_understanding.ipynb`<br>EDA charts in `reports/figures/` |
| **Day 5–7** | **Phase 4: Data Cleaning & Preprocessing** | • Impute median for numeric features, mode/"Unknown" for categoricals<br>• Cap physical outliers (negative visibility, extreme temps)<br>• Purge all post-accident leakage columns (`ID`, `End_Time`, `Description`)<br>• Run 70/15/15 Stratified Split before any transformation | Preprocessor: `src/preprocessor.py`<br>Notebook: `02_preprocessing_and_feature_engineering.ipynb` |

---

## 🗓️ Week 2: Feature Engineering & Baseline Models (Days 7–14)

### Objectives:
- Extract rich domain features from timestamps and weather reports.
- Address class imbalance through multiple paradigms.
- Benchmark 6 diverse machine learning algorithms.

| Day Range | Phase | Key Tasks & Checklists | Deliverable |
| :--- | :--- | :--- | :--- |
| **Day 7–9** | **Phase 5: Feature Engineering** | • Extract temporal indicators: Rush hour, Night/Day, Weekend, Season<br>• Group granular weather into macro classes (Clear, Rain, Fog, Snow)<br>• Derive binary risk triggers (Low Visibility <2mi, Freezing Temp, High Wind)<br>• Encapsulate into reusable `ColumnTransformer` | Module: `src/feature_engineering.py`<br>Fitted pipeline in `models/` |
| **Day 9–10** | **Phase 6: Class Imbalance Handling** | • Implement and compare: (1) `class_weight='balanced'`, (2) SMOTE on train fold, (3) Decision threshold tuning<br>• Demonstrate why raw accuracy fails on minority classes | Benchmark logs comparing SMOTE vs Class Weights |
| **Day 10–14** | **Phase 7: Model Training & Comparison** | • Train 6 architectures: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and MLP Neural Net<br>• Evaluate using Stratified 5-Fold Cross Validation<br>• Tune probability decision thresholds on validation set<br>• Plot side-by-side ROC and Precision-Recall curves | Module: `src/models.py`<br>`reports/model_comparison.csv`<br>`roc_and_pr_curves.png` |

---

## 🗓️ Week 3: Risk Factor Identification & Explainability (Days 15–20)

### Objectives:
- Uncover specific road and environmental conditions driving severe accidents (Phase 8).
- Finalize held-out test evaluation (Phase 9).
- Deploy interactive Streamlit demonstration application (Phase 10).

| Day Range | Phase | Key Tasks & Checklists | Deliverable |
| :--- | :--- | :--- | :--- |
| **Day 15–18** | **Phase 8: Identifying Risk Factors** | • Calculate SHAP values and produce beeswarm & bar plots<br>• Extract interpretable decision rules with shallow trees<br>• Calculate epidemiological Odds Ratios (OR) & 95% Confidence Intervals<br>• Translate findings into plain-language takeaways | Notebook: `04_risk_factor_identification_shap.ipynb`<br>`reports/risk_factors_findings.md`<br>`reports/decision_rules.txt` |
| **Day 18–19** | **Phase 9: Final Test Evaluation** | • Evaluate the champion model on held-out test set (unseen)<br>• Perform error analysis (inspect false negatives in serious cases)<br>• Check model robustness across regions and times of day | `reports/final_test_evaluation.csv`<br>Error analysis report |
| **Day 19–20** | **Phase 10: Streamlit Web Demo** | • Build multi-tab interactive Streamlit web dashboard<br>• Provide real-time slider/dropdown severity risk inference<br>• Display interactive model benchmarks and SHAP visualizations | Interactive app: `app/app.py`<br>Run via `streamlit run app/app.py` |

---

## 🗓️ Week 4: Final Documentation, Report & Viva Defense (Days 21–25)

### Objectives:
- Compile final project report, research slides, and viva defense talking points.

| Day Range | Phase | Key Tasks & Checklists | Deliverable |
| :--- | :--- | :--- | :--- |
| **Day 21–23** | **Phase 11: Comprehensive Report** | • Abstract, Introduction, Literature Review<br>• Methodology & Pipeline Architecture diagram<br>• Benchmark Results table & Error Analysis<br>• Policy recommendations from Odds Ratios | Final Project Thesis / Report (PDF/Word) |
| **Day 23–25** | **Viva & Presentation Prep** | • Prepare 12–15 slide presentation deck<br>• Rehearse live demonstration with Streamlit app<br>• Practice defending against common viva traps (leakage, imbalance, metric choices) | Slide Deck (`presentation.pptx` / PDF)<br>Live demo script |

---

## 🎓 Viva Defense Quick Reference: Common Questions & Winning Answers

1. **Q: Why didn't you just optimize for Accuracy?**
   - **A:** Severe accidents represent only ~20% of incidents. A dummy classifier predicting non-serious every time achieves 80% accuracy but has **0% recall**, completely failing the core goal of identifying life-threatening accidents. We optimized for **Macro F1, Minority Recall, and PR-AUC**.

2. **Q: How did you ensure zero Data Leakage?**
   - **A:** We eliminated all post-incident variables (injury descriptions, emergency response times, road closure durations) and performed the **70/15/15 stratified train/val/test split before** any preprocessing, scaling, or imputation. The `ColumnTransformer` is fitted strictly on the training partition.

3. **Q: What is the real-world value of Phase 8 (Risk Factor Identification)?**
   - **A:** Machine learning prediction tells dispatchers *what* might happen, but SHAP values and Odds Ratios tell urban planners and police *why* it happens. For example, knowing that low visibility under 2 miles increases severity odds by **4.02x** justifies automated variable speed limit signs during heavy fog or storm warnings.
