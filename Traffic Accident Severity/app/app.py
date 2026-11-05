"""
Interactive Streamlit Web Dashboard:
Road Traffic Accident Severity Prediction & Risk Factor Identification.
Includes:
- Real-time accident severity risk inference engine
- Model comparison and benchmark performance visualizer
- Explainable AI (SHAP) & Statistical Odds Ratio risk factor discovery
- Project architecture and viva / demonstration documentation
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sys
import streamlit as st

# Auto-launch via streamlit CLI if run directly with `python app/app.py`
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

# Configure Streamlit Page
st.set_page_config(
    page_title="Accident Severity AI & Risk Factor Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-high {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
    }
    .risk-low {
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #86efac;
        padding: 16px;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PIPELINE_PATH = MODELS_DIR / "best_model_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
ODDS_RATIO_CSV = REPORTS_DIR / "risk_factor_odds_ratios.csv"


@st.cache_resource
def load_trained_pipeline():
    """Loads the pre-trained champion ML pipeline and feature configuration."""
    if PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            return pipeline
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            return None
    return None


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_comparison_data():
    if COMPARISON_CSV.exists():
        return pd.read_csv(COMPARISON_CSV)
    return None


@st.cache_data
def load_odds_ratios():
    if ODDS_RATIO_CSV.exists():
        return pd.read_csv(ODDS_RATIO_CSV)
    return None


pipeline_data = load_trained_pipeline()
metadata = load_metadata()
comparison_df = load_comparison_data()
odds_df = load_odds_ratios()

# Sidebar: Project Overview & Scope
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car.png", width=70)
    st.title("Traffic Safety AI")
    st.caption("Road Accident Severity Prediction & Risk Identification")
    st.divider()

    st.markdown("### 📊 Project Scoping")
    st.markdown(
        """
        - **Target:** Binary Severity (Serious vs Non-Serious)
        - **Primary Dataset:** US Road Accidents
        - **Core Metric:** Macro F1 & Serious Class Recall
        - **Mitigation:** Balanced weights & threshold tuning
        """
    )

    if metadata:
        st.divider()
        st.markdown("### 🏆 Champion Model")
        st.write(f"**Model:** `{metadata.get('best_model_name', 'Trained Classifier')}`")
        st.write(f"**Optimal Threshold:** `{metadata.get('optimal_threshold', 0.5):.2f}`")
        st.write(f"**Test Macro F1:** `{metadata.get('test_macro_f1', 0.0):.3f}`")
        st.write(f"**Serious Recall:** `{metadata.get('test_recall_serious', 0.0):.3f}`")


# Main Dashboard Tabs
tab_pred, tab_models, tab_shap, tab_docs = st.tabs(
    ["🚨 Real-Time Risk Predictor", "📈 Model Benchmarks", "🔍 SHAP & Risk Factors", "📖 Project Methodology"]
)

# ==========================================
# TAB 1: REAL-TIME SEVERITY PREDICTOR
# ==========================================
with tab_pred:
    st.header("Real-Time Accident Severity Predictor")
    st.markdown(
        "Enter environmental, temporal, and road geometry conditions to evaluate the probability of an accident being **Serious / Severe**."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌤️ Weather & Environment")
        weather_cond = st.selectbox(
            "Weather Condition Group",
            ["Clear", "Cloudy", "Rain", "Snow/Ice", "Fog/Haze", "Severe_Storm", "Other"],
            index=0,
        )
        visibility = st.slider("Visibility (miles)", min_value=0.0, max_value=10.0, value=9.5, step=0.5)
        temperature = st.slider("Temperature (°F)", min_value=-20, max_value=115, value=65, step=1)
        humidity = st.slider("Humidity (%)", min_value=5, max_value=100, value=55, step=1)
        wind_speed = st.slider("Wind Speed (mph)", min_value=0, max_value=60, value=8, step=1)
        precipitation = st.slider("Precipitation (inches)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

    with col2:
        st.subheader("⏰ Temporal & Timing")
        hour = st.slider("Hour of Day (24h)", min_value=0, max_value=23, value=17, step=1)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
        month = st.selectbox("Month of Year", list(range(1, 13)), index=9)
        day_night = st.radio("Lighting / Ambient Period", ["Day", "Night"], index=0, horizontal=True)
        highway_side = st.radio("Road / Highway Side", ["Right (R)", "Left (L)"], index=0, horizontal=True)

    with col3:
        st.subheader("🛣️ Road Geometry & Features")
        distance = st.number_input("Traffic Extent / Distance (miles)", min_value=0.0, max_value=15.0, value=0.5, step=0.1)
        st.write("**Road Infrastructure Flags Present:**")
        junction = st.checkbox("Highway Junction / Interchange", value=False)
        traffic_signal = st.checkbox("Traffic Signal Present", value=True)
        crossing = st.checkbox("Pedestrian Crossing", value=False)
        stop_sign = st.checkbox("Stop Sign", value=False)
        station = st.checkbox("Transit / Bus / Train Station", value=False)
        railway = st.checkbox("Railway Crossing", value=False)

    st.write("")
    predict_btn = st.button("🚨 Analyze Accident Severity Risk", type="primary", width="stretch")

    if predict_btn:
        if pipeline_data is None:
            st.warning("⚠️ ML Pipeline model artifacts are currently compiling. Please wait for model training to complete!")
        else:
            preprocessor = pipeline_data["preprocessor"]
            model = pipeline_data["model"]
            opt_thresh = pipeline_data.get("optimal_threshold", 0.50)

            # Compute derived features matching training pipeline
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_num = day_map[day_of_week]
            is_weekend = 1 if day_num in [5, 6] else 0
            is_rush = 1 if ((7 <= hour <= 9 or 16 <= hour <= 18) and is_weekend == 0) else 0
            is_night = 1 if (hour >= 20 or hour < 6) else 0

            # Part of day
            if 5 <= hour < 12:
                day_part = "Morning"
            elif 12 <= hour < 17:
                day_part = "Afternoon"
            elif 17 <= hour < 21:
                day_part = "Evening"
            else:
                day_part = "Night"

            # Season
            if month in [12, 1, 2]:
                season = "Winter"
            elif month in [3, 4, 5]:
                season = "Spring"
            elif month in [6, 7, 8]:
                season = "Summer"
            else:
                season = "Fall"

            side_code = "R" if "Right" in highway_side else "L"

            # Build single-row DataFrame
            sample_dict = {
                "Temperature(F)": float(temperature),
                "Humidity(%)": float(humidity),
                "Pressure(in)": 29.8,
                "Visibility(mi)": float(visibility),
                "Wind_Speed(mph)": float(wind_speed),
                "Precipitation(in)": float(precipitation),
                "Distance(mi)": float(distance),
                "Hour": int(hour),
                "DayOfWeek": int(day_num),
                "Month": int(month),
                "Is_Weekend": int(is_weekend),
                "Is_Rush_Hour": int(is_rush),
                "Is_Night": int(is_night),
                "Low_Visibility": 1 if visibility < 2.0 else 0,
                "Is_Freezing": 1 if temperature <= 32.0 else 0,
                "High_Wind": 1 if wind_speed > 25.0 else 0,
                "Has_Precipitation": 1 if precipitation > 0.0 else 0,
                "Side": side_code,
                "Sunrise_Sunset": day_night,
                "Weather_Condition_Group": weather_cond,
                "Day_Part": day_part,
                "Season": season,
                "Amenity": 0,
                "Bump": 0,
                "Crossing": int(crossing),
                "Give_Way": 0,
                "Junction": int(junction),
                "No_Exit": 0,
                "Railway": int(railway),
                "Roundabout": 0,
                "Station": int(station),
                "Stop": int(stop_sign),
                "Traffic_Calming": 0,
                "Traffic_Signal": int(traffic_signal),
            }

            input_df = pd.DataFrame([sample_dict])

            try:
                # Transform using preprocessor
                input_proc = preprocessor.transform(input_df)

                # Predict probability
                if hasattr(model, "predict_proba"):
                    prob_serious = model.predict_proba(input_proc)[0, 1]
                else:
                    prob_serious = float(model.predict(input_proc)[0])

                is_pred_serious = prob_serious >= opt_thresh

                st.divider()
                res_col1, res_col2 = st.columns([1, 1])

                with res_col1:
                    st.subheader("Prediction Result")
                    if is_pred_serious:
                        st.markdown(
                            f"""
                            <div class="risk-high">
                                <h3>⚠️ High Risk: SERIOUS ACCIDENT</h3>
                                <p>The model predicts this incident has a high probability of severe impact, major road blockage, or serious casualties.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="risk-low">
                                <h3>✅ Low / Moderate Risk: NON-SERIOUS</h3>
                                <p>The model predicts this incident is likely minor or moderate (short delay, property damage).</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_col2:
                    st.subheader("Severity Probability Analysis")
                    st.write(f"**Predicted Serious Probability:** `{prob_serious * 100:.1f}%`")
                    st.write(f"**Decision Threshold:** `{opt_thresh * 100:.1f}%` (Optimized on Validation Set)")
                    st.progress(float(prob_serious))

                    # High risk flags triggered
                    risk_factors_triggered = []
                    if visibility < 2.0:
                        risk_factors_triggered.append("Low Visibility (< 2 mi)")
                    if is_night == 1:
                        risk_factors_triggered.append("Nighttime driving conditions")
                    if junction == 1:
                        risk_factors_triggered.append("Highway Junction / Interchange")
                    if weather_cond in ["Rain", "Snow/Ice", "Fog/Haze", "Severe_Storm"]:
                        risk_factors_triggered.append(f"Adverse Weather ({weather_cond})")
                    if temperature <= 32.0:
                        risk_factors_triggered.append("Freezing temperatures (Icy road hazard)")

                    if risk_factors_triggered:
                        st.info("⚠️ **Active Risk Triggers Detected:** " + ", ".join(risk_factors_triggered))
                    else:
                        st.success("✨ No extreme hazard conditions detected for this profile.")

            except Exception as ex:
                st.error(f"Inference error: {ex}")


# ==========================================
# TAB 2: MODEL BENCHMARKS & EVALUATION
# ==========================================
with tab_models:
    st.header("Phase 7: Model Benchmarks & Comparison")
    st.markdown(
        "Comparison of 6 diverse model architectures on stratified validation data. Imbalance mitigated through class weighting and threshold tuning."
    )

    if comparison_df is not None:
        st.dataframe(comparison_df.style.highlight_max(subset=["Macro F1", "Recall (Serious)", "ROC-AUC", "PR-AUC"], color="#1e3a8a"), width="stretch")

        roc_pr_img = FIGURES_DIR / "roc_and_pr_curves.png"
        cm_img = FIGURES_DIR / "confusion_matrices_top_models.png"

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if roc_pr_img.exists():
                st.image(str(roc_pr_img), caption="ROC & Precision-Recall Curves across candidate models")
        with col_img2:
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrices for top ensemble models")
    else:
        st.info("Training comparison metrics will automatically appear here once training completes.")


# ==========================================
# TAB 3: SHAP & RISK FACTOR IDENTIFICATION
# ==========================================
with tab_shap:
    st.header("Phase 8: Identifying Conditions Linked to Serious Accidents")
    st.markdown(
        "Using **SHAP (SHapley Additive exPlanations)** and **Epidemiological Odds Ratios (OR)** to discover which conditions systematically push severity up."
    )

    if odds_df is not None:
        st.subheader("📊 Statistical Odds Ratios (Relative Risk of Severe Outcomes)")
        st.dataframe(odds_df, width="stretch")

        st.subheader("💡 Plain-Language Safety Findings")
        for statement in odds_df["Plain-Language Finding"]:
            st.markdown(f"- {statement}")

    shap_beeswarm = FIGURES_DIR / "shap_summary_beeswarm.png"
    shap_bar = FIGURES_DIR / "shap_importance_bar.png"

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if shap_beeswarm.exists():
            st.image(str(shap_beeswarm), caption="SHAP Summary Beeswarm Plot: Feature Impact on Serious Class")
    with col_s2:
        if shap_bar.exists():
            st.image(str(shap_bar), caption="Global Feature Importance (Mean |SHAP Value|)")

    decision_rules_file = REPORTS_DIR / "decision_rules.txt"
    if decision_rules_file.exists():
        st.subheader("🌳 Extracted Decision Rules (Shallow Tree)")
        with open(decision_rules_file, "r") as f:
            st.code(f.read(), language="text")


# ==========================================
# TAB 4: PROJECT METHODOLOGY & SCOPING
# ==========================================
with tab_docs:
    st.header("Project Architecture & Defense Guide")
    st.markdown(
        """
        ### 🎯 Problem Statement & Scoping
        - **Objective:** Predict road traffic accident severity and systematically identify environmental and road conditions linked to serious incidents.
        - **Target Formulation:** Binary classification where `Severity >= 3` denotes Serious/Fatal accidents requiring emergency dispatch.
        - **Metric Priority:** Accuracy is misleading due to severe class imbalance (~15–20% serious). We evaluate on **Macro F1, Minority Recall, and PR-AUC**.

        ### 🛡️ Preventing Data Leakage
        1. All post-incident attributes (casualty counts, injury descriptions, civil twilight endpoints) are strictly purged.
        2. Train/Validation/Test split (70/15/15) is executed **before** scaling, imputation, or resampling.
        3. Preprocessing `ColumnTransformer` is fitted solely on the training partition.
        """
    )
