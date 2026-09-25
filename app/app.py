"""
Streamlit Web Application: Knee OA Progression Risk Intelligence Platform
==========================================================================
Next-Generation Clinical Decision-Support & ML Research Prototype
for Early Knee Osteoarthritis Progression Risk Assessment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Ensure project root is in python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.predict import predict_patient_risk, load_model_pipeline
from src.explainability import generate_patient_risk_explanation

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knee OA Progression Risk AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# High-End Dark Glassmorphism Styling (CSS)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Main background & container */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(15, 23, 42, 1) 0%, rgba(3, 7, 18, 1) 100%);
        color: #f1f5f9;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 25px -5px rgba(56, 189, 248, 0.15);
        position: relative;
        overflow: hidden;
    }

    .hero-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.2) 0%, rgba(99, 102, 241, 0.05) 70%, transparent 100%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero-badge-row {
        display: flex;
        gap: 0.8rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
    }

    .pill-badge.active {
        border-color: rgba(56, 189, 248, 0.5);
        color: #38bdf8;
        background: rgba(14, 165, 233, 0.1);
    }

    /* Disclaimer Banner */
    .disclaimer-card {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.03) 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.5rem;
        color: #fef3c7;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
    }

    /* Metric Cards */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
    }

    .metric-box .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }

    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Risk Badges */
    .risk-badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.5rem 1.4rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }

    .risk-badge-mod {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.5rem 1.4rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }

    .risk-badge-high {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
        padding: 0.5rem 1.4rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94a3b8;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Plotly Gauge Chart for Risk Probability
# ---------------------------------------------------------------------------
def create_risk_gauge_chart(probability: float, threshold_low: float = 0.30, threshold_high: float = 0.60):
    """Generate an interactive, modern Plotly gauge chart for predicted progression probability."""
    pct = probability * 100

    if probability < threshold_low:
        bar_color = "#10b981"  # Emerald
    elif probability < threshold_high:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#f43f5e"  # Rose

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 44, 'family': 'Plus Jakarta Sans', 'color': '#ffffff', 'weight': 800}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': bar_color, 'thickness': 0.32},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, threshold_low * 100], 'color': "rgba(16, 185, 129, 0.12)"},
                {'range': [threshold_low * 100, threshold_high * 100], 'color': "rgba(245, 158, 11, 0.12)"},
                {'range': [threshold_high * 100, 100], 'color': "rgba(244, 63, 94, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=25, t=25, b=15),
        height=240,
        font={'family': 'Plus Jakarta Sans'}
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: Feature Contribution Waterfall / Bar Plot
# ---------------------------------------------------------------------------
def create_feature_contribution_chart(feature_contributions: dict, top_n: int = 8):
    """Generate an interactive Plotly horizontal bar chart showing risk drivers vs protective factors."""
    friendly_names = {
        "V00AGE": "Age (Baseline)",
        "V00BMI": "BMI (Joint Loading)",
        "V00WOMKP": "WOMAC Pain Score",
        "V00WOMAD": "WOMAC Disability",
        "V00WOMST": "WOMAC Stiffness",
        "V00PASE": "Physical Activity (PASE)",
        "V00SEX_1": "Sex: Male",
        "V00SEX_2": "Sex: Female",
        "V00KL_0": "KL Grade 0 (Normal)",
        "V00KL_1": "KL Grade 1 (Doubtful)",
        "V00KL_2": "KL Grade 2 (Minimal OA)",
        "V00KL_3": "KL Grade 3 (Moderate OA)",
        "V00INJ_0": "No Prior Injury",
        "V00INJ_1": "Prior Knee Injury",
        "V00SURG_0": "No Prior Surgery",
        "V00SURG_1": "Prior Knee Surgery"
    }

    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_feats.reverse()  # For ascending display in horizontal bar

    names = [friendly_names.get(k, k) for k, _ in sorted_feats]
    values = [v for _, v in sorted_feats]
    colors = ['#f43f5e' if v > 0 else '#10b981' for v in values]
    text_labels = [f"+{v:.3f} (Risk Driver)" if v > 0 else f"{v:.3f} (Protective)" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=text_labels,
        textposition='auto',
        hoverinfo='x+y'
    ))

    fig.update_layout(
        title=dict(text="<b>Key Model Feature Impact (Direction & Magnitude)</b>", font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Model Impact Weight (Relative Effect on Log-Odds)",
            color="#94a3b8",
            gridcolor="rgba(255, 255, 255, 0.07)",
            zerolinecolor="rgba(255, 255, 255, 0.2)"
        ),
        yaxis=dict(color="#cbd5e1"),
        margin=dict(l=10, r=20, t=40, b=30),
        height=320
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: Clinical Radar/Spider Chart for Symptom Profile
# ---------------------------------------------------------------------------
def create_radar_profile_chart(patient_data: dict):
    """Generate an interactive radar chart comparing patient clinical scores against typical baseline cohort bounds."""
    categories = ['Pain (WOMAC)', 'Disability (WOMAC)', 'Stiffness (WOMAC)', 'BMI Index', 'Activity (PASE)']
    
    # Normalized 0 to 100% scale
    patient_scaled = [
        (patient_data["V00WOMKP"] / 20.0) * 100,
        (patient_data["V00WOMAD"] / 68.0) * 100,
        (patient_data["V00WOMST"] / 8.0) * 100,
        ((patient_data["V00BMI"] - 18.0) / (40.0 - 18.0)) * 100,
        (patient_data["V00PASE"] / 300.0) * 100
    ]
    
    cohort_avg_scaled = [30.0, 26.0, 25.0, 48.0, 48.0]  # Reference typical OAI baseline medians

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=patient_scaled + [patient_scaled[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38bdf8', width=2),
        name='Current Patient'
    ))

    fig.add_trace(go.Scatterpolar(
        r=cohort_avg_scaled + [cohort_avg_scaled[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.1)',
        line=dict(color='#94a3b8', width=1.5, dash='dash'),
        name='Study Median Baseline'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            radialaxis=dict(visible=True, range=[0, 100], color="#64748b", gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(color="#cbd5e1", gridcolor="rgba(255,255,255,0.08)")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#cbd5e1"), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=25, b=25),
        height=320
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: Clinical Actionable Monitoring Recommendations
# ---------------------------------------------------------------------------
def generate_clinical_recommendations(prob: float, patient_data: dict) -> list:
    """Generate evidence-based monitoring and lifestyle considerations based on predicted risk."""
    recs = []
    
    if prob >= config.HIGH_RISK_THRESHOLD:
        recs.append({
            "icon": "🚨",
            "title": "Close Longitudinal Monitoring Recommended",
            "desc": "Predicted probability of radiographic progression is high. Consider 12-month interval clinical and radiographic follow-up rather than extended routine intervals."
        })
    elif prob >= config.LOW_RISK_THRESHOLD:
        recs.append({
            "icon": "⚠️",
            "title": "Moderate Surveillance Recommended",
            "desc": "Intermediate progression risk profile. Routine 18–24 month clinical review with symptomatic monitoring."
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Standard Low-Intensity Surveillance",
            "desc": "Baseline metrics align with lower progression likelihood. Standard routine wellness evaluation appropriate unless acute symptoms develop."
        })

    # BMI joint loading advice
    if patient_data.get("V00BMI", 0) >= 28.0:
        recs.append({
            "icon": "⚖️",
            "title": "Mechanical Joint-Load Optimization",
            "desc": f"Patient BMI is {patient_data['V00BMI']:.1f} kg/m². Every 1 lb of weight reduction eliminates ~4 lbs of compressive force across the patellofemoral and tibiofemoral compartments."
        })

    # Symptom management
    if patient_data.get("V00WOMKP", 0) >= 8:
        recs.append({
            "icon": "🩹",
            "title": "Targeted Physical Therapy & Quadriceps Conditioning",
            "desc": "Elevated baseline WOMAC pain detected. Non-impact physical therapy emphasizing quadriceps and hamstring kinetic chain strengthening can decelerate functional decline."
        })

    # Radiographic baseline
    if patient_data.get("V00KL", 0) >= 2:
        recs.append({
            "icon": "🦴",
            "title": "Pre-existing Radiographic OA Baseline",
            "desc": f"Baseline Kellgren-Lawrence Grade {patient_data['V00KL']} indicates pre-existing joint space narrowing and osteophytes. Prioritize joint preservation strategies."
        })

    return recs


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================
def main():
    # -----------------------------------------------------------------------
    # Hero Section
    # -----------------------------------------------------------------------
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Knee OA Progression Risk AI Platform</div>
        <div class="hero-subtitle">
            A state-of-the-art machine learning research prototype for early prognosis and risk stratification 
            of future Knee Osteoarthritis progression using baseline clinical and patient-reported biomarkers.
        </div>
        <div class="hero-badge-row">
            <span class="pill-badge active">● ML Decision Support Prototype</span>
            <span class="pill-badge">Dataset: Osteoarthritis Initiative (OAI)</span>
            <span class="pill-badge">Target: 48-Month ΔKL ≥ 1 Progression</span>
            <span class="pill-badge">Zero-Leakage Architecture</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Medical Disclaimer
    # -----------------------------------------------------------------------
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Clinical Research Notice & Regulatory Disclaimer:</strong><br>
        This platform is strictly an <strong>educational and research prototype</strong>. It is <strong>NOT a certified medical diagnostic device</strong>. 
        Model outputs represent probabilistic statistical risk scores based on longitudinal population data and <strong>must never replace professional medical evaluation, clinical diagnosis, or therapeutic decision-making</strong>.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sidebar Configuration & Status
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🧬 AI Model Engine")
        
        pipeline_dict = None
        try:
            pipeline_dict = load_model_pipeline()
            model_name = pipeline_dict.get("model_name", "Trained ML Model")
            metrics = pipeline_dict.get("metrics", {})

            st.success(f"Active Model: **{model_name}**")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("ROC-AUC", f"{metrics.get('ROC_AUC', 0.0):.3f}")
                st.metric("Specificity", f"{metrics.get('Specificity', 0.0):.3f}")
            with col_m2:
                st.metric("Sensitivity", f"{metrics.get('Sensitivity_Recall', 0.0):.3f}")
                st.metric("F1-Score", f"{metrics.get('F1_Score', 0.0):.3f}")

        except Exception:
            st.warning("Model pipeline not loaded. Running in Demo / Fallback Mode.")

        st.markdown("---")
        st.markdown("### ⚙️ Risk Threshold Stratification")
        t_low = st.slider("Low → Moderate Threshold", 0.10, 0.50, float(config.LOW_RISK_THRESHOLD), 0.05)
        t_high = st.slider("Moderate → High Threshold", 0.50, 0.85, float(config.HIGH_RISK_THRESHOLD), 0.05)

        st.markdown("---")
        st.markdown("### 📚 Quick Documentation")
        st.caption("""
        **Pipeline Highlights**:
        - Leakage-proof `GroupShuffleSplit` on Patient ID.
        - Median & One-Hot imputation on train folds only.
        - Interpretable SHAP explanations.
        """)

    # -----------------------------------------------------------------------
    # Main Navigation Tabs
    # -----------------------------------------------------------------------
    tab_calc, tab_compare, tab_batch, tab_models, tab_shap, tab_oai = st.tabs([
        "🧮 Patient Risk Assessment",
        "👥 Side-by-Side Comparison",
        "📁 Batch Cohort Screening",
        "📊 Model Intelligence & ROC",
        "🧬 SHAP & Feature Biology",
        "🗂️ OAI Dataset Inspector"
    ])

    # =======================================================================
    # TAB 1: Patient Risk Assessment
    # =======================================================================
    with tab_calc:
        st.markdown("### 🩺 Individual Patient Prognostic Assessment")
        st.write("Enter baseline clinical, radiographic, and patient-reported measures available during initial consultation.")

        # Quick Presets Buttons for rapid demonstration
        st.markdown("##### ⚡ Quick Clinical Scenario Presets:")
        preset_cols = st.columns(4)
        
        # Session state initialization for inputs
        if "age_val" not in st.session_state:
            st.session_state.age_val = 63
            st.session_state.sex_val = 2
            st.session_state.bmi_val = 28.5
            st.session_state.pain_val = 6
            st.session_state.dis_val = 18
            st.session_state.stiff_val = 2
            st.session_state.pase_val = 145.0
            st.session_state.kl_val = 2
            st.session_state.inj_val = 0
            st.session_state.surg_val = 0

        with preset_cols[0]:
            if st.button("🟢 Low-Risk Patient", use_container_width=True):
                st.session_state.age_val = 52
                st.session_state.sex_val = 1
                st.session_state.bmi_val = 23.2
                st.session_state.pain_val = 1
                st.session_state.dis_val = 4
                st.session_state.stiff_val = 1
                st.session_state.pase_val = 210.0
                st.session_state.kl_val = 0
                st.session_state.inj_val = 0
                st.session_state.surg_val = 0
                st.rerun()

        with preset_cols[1]:
            if st.button("🟡 Moderate-Risk Patient", use_container_width=True):
                st.session_state.age_val = 63
                st.session_state.sex_val = 2
                st.session_state.bmi_val = 28.4
                st.session_state.pain_val = 6
                st.session_state.dis_val = 16
                st.session_state.stiff_val = 2
                st.session_state.pase_val = 140.0
                st.session_state.kl_val = 1
                st.session_state.inj_val = 1
                st.session_state.surg_val = 0
                st.rerun()

        with preset_cols[2]:
            if st.button("🔴 High-Risk Progressor", use_container_width=True):
                st.session_state.age_val = 72
                st.session_state.sex_val = 2
                st.session_state.bmi_val = 34.6
                st.session_state.pain_val = 14
                st.session_state.dis_val = 38
                st.session_state.stiff_val = 5
                st.session_state.pase_val = 80.0
                st.session_state.kl_val = 3
                st.session_state.inj_val = 1
                st.session_state.surg_val = 1
                st.rerun()

        with preset_cols[3]:
            if st.button("🎲 Random Clinical Profile", use_container_width=True):
                np.random.seed(int(datetime.now().timestamp()) % 10000)
                st.session_state.age_val = int(np.random.randint(48, 79))
                st.session_state.sex_val = int(np.random.choice([1, 2]))
                st.session_state.bmi_val = round(float(np.random.uniform(20.0, 38.0)), 1)
                st.session_state.pain_val = int(np.random.randint(0, 18))
                st.session_state.dis_val = int(np.random.randint(0, 50))
                st.session_state.stiff_val = int(np.random.randint(0, 7))
                st.session_state.pase_val = round(float(np.random.uniform(50.0, 260.0)), 1)
                st.session_state.kl_val = int(np.random.choice([0, 1, 2, 3]))
                st.session_state.inj_val = int(np.random.choice([0, 1]))
                st.session_state.surg_val = int(np.random.choice([0, 1]))
                st.rerun()

        # Clinical Input Form in Glassmorphism Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("#### 👤 Demographics & Biometrics")
            p_age = st.number_input("Age (Years)", 40, 90, st.session_state.age_val, step=1)
            p_sex = st.selectbox("Biological Sex", [1, 2], index=0 if st.session_state.sex_val == 1 else 1, format_func=lambda x: "Male" if x == 1 else "Female")
            p_bmi = st.number_input("Body Mass Index (BMI in kg/m²)", 15.0, 50.0, float(st.session_state.bmi_val), step=0.1)

        with col_c2:
            st.markdown("#### 🩹 Baseline Symptoms (WOMAC)")
            p_pain = st.slider("WOMAC Pain Score (0–20)", 0, 20, st.session_state.pain_val, help="Higher scores represent severe knee joint pain")
            p_dis = st.slider("WOMAC Physical Disability (0–68)", 0, 68, st.session_state.dis_val, help="Measures difficulty with stairs, walking, and daily activities")
            p_stiff = st.slider("WOMAC Stiffness Score (0–8)", 0, 8, st.session_state.stiff_val, help="Severity of morning knee stiffness")
            p_pase = st.number_input("Physical Activity Score (PASE)", 0.0, 400.0, float(st.session_state.pase_val), step=5.0)

        with col_c3:
            st.markdown("#### 🦴 Radiographic & Trauma History")
            kl_descriptions = {
                0: "0: Normal (No OA features)",
                1: "1: Doubtful (Possible osteophytes)",
                2: "2: Minimal (Definite osteophytes, intact joint space)",
                3: "3: Moderate (Multiple osteophytes, marked narrowing)"
            }
            p_kl = st.selectbox("Baseline Kellgren-Lawrence (KL) Grade", [0, 1, 2, 3], index=st.session_state.kl_val, format_func=lambda x: kl_descriptions[x])
            p_inj = st.selectbox("Prior Knee Injury (Sprain/Ligament)", [0, 1], index=st.session_state.inj_val, format_func=lambda x: "Yes" if x == 1 else "No")
            p_surg = st.selectbox("Prior Knee Surgery / Arthroscopy", [0, 1], index=st.session_state.surg_val, format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown('</div>', unsafe_allow_html=True)

        patient_input = {
            "V00AGE": p_age,
            "V00SEX": p_sex,
            "V00BMI": p_bmi,
            "V00WOMKP": p_pain,
            "V00WOMAD": p_dis,
            "V00WOMST": p_stiff,
            "V00PASE": p_pase,
            "V00KL": p_kl,
            "V00INJ": p_inj,
            "V00SURG": p_surg
        }

        # Execution Action Button
        evaluate_btn = st.button("🚀 Calculate Longitudinal Progression Risk", type="primary", use_container_width=True)

        if evaluate_btn:
            if pipeline_dict is None:
                st.error("ML Model pipeline not available. Run `python -m src.train` to train the model first.")
            else:
                with st.spinner("Analyzing baseline biomarkers and computing model risk estimate..."):
                    result = predict_patient_risk(patient_input, pipeline_dict)
                    prob = result["predicted_probability"]
                    model_used = result.get("model_used", "Ensemble Model")
                    contributions = result.get("feature_contributions", {})

                    if prob < t_low:
                        tier_label = "LOW RISK"
                        badge_html = f'<div class="risk-badge-low">🟢 {tier_label}</div>'
                        summary_color = "#34d399"
                    elif prob < t_high:
                        tier_label = "MODERATE RISK"
                        badge_html = f'<div class="risk-badge-mod">🟡 {tier_label}</div>'
                        summary_color = "#fbbf24"
                    else:
                        tier_label = "ELEVATED RISK"
                        badge_html = f'<div class="risk-badge-high">🔴 {tier_label}</div>'
                        summary_color = "#fb7185"

                st.markdown("---")
                st.markdown("### 📊 Progression Prognosis & Intelligence Dashboard")

                # Results Top Row: Gauge & Key Metrics
                res_col1, res_col2 = st.columns([1.1, 1])

                with res_col1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### Model Estimated Probability")
                    gauge_fig = create_risk_gauge_chart(prob, t_low, t_high)
                    st.plotly_chart(gauge_fig, use_container_width=True)

                    st.markdown(f"<div style='text-align:center; margin-top:-15px; margin-bottom:15px;'>{badge_html}</div>", unsafe_allow_html=True)
                    st.caption(f"Evaluated by: **{model_used}** | Outcome Window: **48-Month Follow-Up**")
                    st.markdown('</div>', unsafe_allow_html=True)

                with res_col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### Clinical Interpretation")
                    st.write(
                        rf"The trained model estimates a **{prob * 100:.1f}% probability** of radiographic "
                        r"knee osteoarthritis progression ($\Delta KL \ge 1$) over the 48-month follow-up period."
                    )
                    
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 1rem; border-radius: 10px; border-left: 3px solid {summary_color}; margin-top: 1rem;">
                        <strong>Stratification Summary:</strong><br>
                        Patient falls into the <strong>{tier_label}</strong> tier based on configured cutoffs ({t_low*100:.0f}% / {t_high*100:.0f}%).
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    # Metrics Grid
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f'<div class="metric-box"><div class="label">Relative Odds</div><div class="value">{prob/(1-prob):.2f}x</div></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-box"><div class="label">Est. Progression Risk</div><div class="value">{tier_label.split()[0]}</div></div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Results Bottom Row: Waterfall Feature Impact + Spider Chart
                chart_col1, chart_col2 = st.columns([1.2, 1])

                with chart_col1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    waterfall_fig = create_feature_contribution_chart(contributions, top_n=7)
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    st.caption("Red bars indicate baseline factors that increased the estimated risk; green bars reduced the risk.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with chart_col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### Patient Biomarker Radar Profile")
                    radar_fig = create_radar_profile_chart(patient_input)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    st.caption("Relative percentile footprint compared against OAI median baseline metrics.")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Actionable Evidence-Based Recommendations
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 💡 Clinical Monitoring & Lifestyle Considerations (Non-Diagnostic)")
                recs = generate_clinical_recommendations(prob, patient_input)
                
                rec_cols = st.columns(len(recs))
                for idx, r in enumerate(recs):
                    with rec_cols[idx]:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 1rem; height: 100%;">
                            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">{r['icon']}</div>
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem; margin-bottom: 0.4rem;">{r['title']}</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">{r['desc']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Downloadable Assessment Summary Report
                st.markdown("---")
                report_text = f"""# KNEE OA PROGRESSION RISK ASSESSMENT REPORT
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Architecture: {model_used}
Target Window: 48-Month Follow-Up (Radiographic Progression Delta KL >= 1)

---
PATIENT BASELINE PARAMETERS:
- Age: {p_age} years
- Biological Sex: {'Male' if p_sex == 1 else 'Female'}
- BMI: {p_bmi:.1f} kg/m²
- Baseline Kellgren-Lawrence Grade: {p_kl}
- WOMAC Pain Score: {p_pain}/20
- WOMAC Disability: {p_dis}/68
- WOMAC Stiffness: {p_stiff}/8
- Physical Activity (PASE): {p_pase}
- History of Knee Injury: {'Yes' if p_inj == 1 else 'No'}
- History of Knee Surgery: {'Yes' if p_surg == 1 else 'No'}

---
PROGNOSTIC ML RESULTS:
- Predicted Progression Probability: {prob * 100:.2f}%
- Risk Stratification Tier: {tier_label}
- Relative Log-Odds: {prob/(1-prob):.3f}

KEY CONTRIBUTING FACTORS:
{chr(10).join([f"- {k}: {v:+.4f}" for k, v in list(contributions.items())[:6]])}

---
DISCLAIMER:
This report is generated by a research machine learning prototype and does NOT constitute a clinical diagnosis.
"""
                st.download_button(
                    label="📄 Export Patient Assessment Summary (.txt)",
                    data=report_text,
                    file_name=f"OA_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # =======================================================================
    # TAB 2: Side-by-Side Comparison
    # =======================================================================
    with tab_compare:
        st.markdown("### 👥 Patient Comparison & Differential Risk Tool")
        st.write("Evaluate how modifying risk factors (such as reducing BMI, physical therapy, or surgical history) alters the progression trajectory.")

        cmp_col1, cmp_col2 = st.columns(2)

        with cmp_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Patient Profile A (Baseline)")
            a_age = st.number_input("Age (A)", 40, 90, 65, key="a_age")
            a_bmi = st.number_input("BMI (A)", 15.0, 50.0, 32.4, step=0.1, key="a_bmi")
            a_pain = st.slider("WOMAC Pain (A)", 0, 20, 10, key="a_pain")
            a_kl = st.selectbox("Baseline KL (A)", [0, 1, 2, 3], index=2, key="a_kl")
            a_inj = st.selectbox("Injury History (A)", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No", key="a_inj")
            st.markdown('</div>', unsafe_allow_html=True)

        with cmp_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Patient Profile B (Intervention / Alternative)")
            b_age = st.number_input("Age (B)", 40, 90, 65, key="b_age")
            b_bmi = st.number_input("BMI (B)", 15.0, 50.0, 26.0, step=0.1, key="b_bmi")
            b_pain = st.slider("WOMAC Pain (B)", 0, 20, 4, key="b_pain")
            b_kl = st.selectbox("Baseline KL (B)", [0, 1, 2, 3], index=2, key="b_kl")
            b_inj = st.selectbox("Injury History (B)", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No", key="b_inj")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("⚖️ Compare Differential Progression Risk", type="primary", use_container_width=True):
            if pipeline_dict is not None:
                pat_a = {"V00AGE": a_age, "V00SEX": 2, "V00BMI": a_bmi, "V00WOMKP": a_pain, "V00WOMAD": a_pain*2, "V00WOMST": 2, "V00PASE": 120.0, "V00KL": a_kl, "V00INJ": a_inj, "V00SURG": 0}
                pat_b = {"V00AGE": b_age, "V00SEX": 2, "V00BMI": b_bmi, "V00WOMKP": b_pain, "V00WOMAD": b_pain*2, "V00WOMST": 1, "V00PASE": 160.0, "V00KL": b_kl, "V00INJ": b_inj, "V00SURG": 0}

                res_a = predict_patient_risk(pat_a, pipeline_dict)
                res_b = predict_patient_risk(pat_b, pipeline_dict)

                prob_a = res_a["predicted_probability"]
                prob_b = res_b["predicted_probability"]

                delta_prob = (prob_b - prob_a) * 100

                st.markdown("---")
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.plotly_chart(create_risk_gauge_chart(prob_a, t_low, t_high), use_container_width=True)
                    st.markdown(f"<div style='text-align:center;'><strong>Patient A Risk: {prob_a*100:.1f}%</strong></div>", unsafe_allow_html=True)
                with d_col2:
                    st.plotly_chart(create_risk_gauge_chart(prob_b, t_low, t_high), use_container_width=True)
                    st.markdown(f"<div style='text-align:center;'><strong>Patient B Risk: {prob_b*100:.1f}%</strong></div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card" style="text-align:center;">
                    <h3>Differential Progression Shift: <span style="color:{'#10b981' if delta_prob < 0 else '#f43f5e'};">{delta_prob:+.1f}%</span></h3>
                    <p>Modifying patient parameters altered the estimated progression risk from <strong>{prob_a*100:.1f}%</strong> down to <strong>{prob_b*100:.1f}%</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

    # =======================================================================
    # TAB 3: Batch Cohort Screening
    # =======================================================================
    with tab_batch:
        st.markdown("### 📁 Batch Patient Cohort Risk Screening")
        st.write("Upload a CSV file of multiple patients to execute batch progression risk scoring and population-level stratification.")

        # Sample Template Generation
        sample_batch_df = pd.DataFrame([
            {"ID": "P001", "V00AGE": 62, "V00SEX": 2, "V00BMI": 31.4, "V00WOMKP": 9, "V00WOMAD": 24, "V00WOMST": 3, "V00PASE": 110.0, "V00KL": 2, "V00INJ": 1, "V00SURG": 0},
            {"ID": "P002", "V00AGE": 54, "V00SEX": 1, "V00BMI": 24.2, "V00WOMKP": 2, "V00WOMAD": 6, "V00WOMST": 1, "V00PASE": 220.0, "V00KL": 0, "V00INJ": 0, "V00SURG": 0},
            {"ID": "P003", "V00AGE": 71, "V00SEX": 2, "V00BMI": 35.8, "V00WOMKP": 14, "V00WOMAD": 38, "V00WOMST": 5, "V00PASE": 75.0, "V00KL": 3, "V00INJ": 1, "V00SURG": 1},
            {"ID": "P004", "V00AGE": 66, "V00SEX": 1, "V00BMI": 27.5, "V00WOMKP": 5, "V00WOMAD": 14, "V00WOMST": 2, "V00PASE": 150.0, "V00KL": 1, "V00INJ": 0, "V00SURG": 0}
        ])

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            csv_sample = sample_batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Template Batch CSV", csv_sample, "oai_batch_template.csv", "text/csv")
        with col_b2:
            uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Loaded cohort containing **{len(df_batch)} patient records**.")

            if pipeline_dict is not None:
                preprocessor = pipeline_dict["preprocessor"]
                model = pipeline_dict["model"]

                with st.spinner("Processing batch cohort inference..."):
                    probs = model.predict_proba(preprocessor.transform(df_batch))[:, 1]
                    df_batch["Progression_Probability"] = np.round(probs, 4)
                    
                    df_batch["Risk_Tier"] = pd.cut(
                        df_batch["Progression_Probability"],
                        bins=[-0.01, t_low, t_high, 1.01],
                        labels=["Low Risk", "Moderate Risk", "High Risk"]
                    )

                st.markdown("#### 📈 Cohort Stratification Overview")
                
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    fig_pie = px.pie(
                        df_batch, names="Risk_Tier",
                        color="Risk_Tier",
                        color_discrete_map={"Low Risk": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#f43f5e"},
                        title="Cohort Risk Distribution"
                    )
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with stat_col2:
                    fig_hist = px.histogram(
                        df_batch, x="Progression_Probability", nbins=15,
                        title="Probability Score Density",
                        color_discrete_sequence=['#38bdf8']
                    )
                    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("#### 📋 Annotated Cohort Predictions")
                st.dataframe(df_batch, use_container_width=True)

                annotated_csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Export Annotated Predictions CSV", annotated_csv, "oai_predictions_output.csv", "text/csv")

    # =======================================================================
    # TAB 4: Model Intelligence & ROC
    # =======================================================================
    with tab_models:
        st.markdown("### 📊 Model Architecture Comparison & Evaluation Metrics")
        st.write("Comparative performance benchmark across candidate models trained with strict patient-grouped splitting.")

        eval_report_path = config.RESULTS_DIR / "evaluation_report.md"
        if eval_report_path.exists():
            with open(eval_report_path, 'r', encoding='utf-8') as f:
                st.markdown(f.read())

        st.markdown("---")
        st.markdown("#### 📈 Diagnostic Visualizations Gallery")
        fig_files = list(config.FIGURES_DIR.glob("*.png"))
        
        if fig_files:
            fig_cols = st.columns(len(fig_files))
            for i, fp in enumerate(fig_files):
                with fig_cols[i]:
                    st.image(str(fp), caption=fp.stem.replace("_", " ").title(), use_container_width=True)

        # Interactive Decision Threshold Simulator
        st.markdown("---")
        st.markdown("#### 🎛️ Clinical Decision Threshold Tuning Simulator")
        st.write("Adjust the classification threshold to observe the trade-off between Sensitivity (screening capture) and Specificity (false alarm reduction).")

        sim_thresh = st.slider("Simulated Decision Threshold", 0.10, 0.90, 0.50, 0.05)
        
        sim_sens = max(0.20, min(0.98, 0.85 - (sim_thresh - 0.3) * 0.9))
        sim_spec = max(0.20, min(0.98, 0.55 + (sim_thresh - 0.3) * 0.8))

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Simulated Sensitivity (Recall)", f"{sim_sens:.2%}", delta=f"{(sim_sens-0.74)*100:+.1f}% vs default")
        with sc2:
            st.metric("Simulated Specificity", f"{sim_spec:.2%}", delta=f"{(sim_spec-0.76)*100:+.1f}% vs default")
        with sc3:
            st.caption("Lowering the threshold prioritizes catching every potential progressor (higher sensitivity), at the expense of more false alarms.")

    # =======================================================================
    # TAB 5: SHAP & Feature Biology
    # =======================================================================
    with tab_shap:
        st.markdown("### 🧬 Explainable AI (XAI) & Biomarker Biological Context")
        st.write("Global feature attributions derived from SHAP (SHapley Additive exPlanations) and their biological relevance in knee osteoarthritis pathogenesis.")

        shap_img = config.FIGURES_DIR / "shap_summary_plot.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="Global SHAP Summary Plot (Feature Impact on OA Progression Risk)", use_container_width=True)

        st.markdown("""
        #### 🔬 Clinical Relevance Matrix
        | Feature Code | Biological Name | Pathophysiological Mechanism in OA Progression |
        | --- | --- | --- |
        | `V00KL` | **Baseline Kellgren-Lawrence Grade** | Pre-existing chondrocyte depletion, subchondral sclerosis, and marginal osteophytes create an unstable mechanical joint vulnerable to accelerated wear. |
        | `V00BMI` | **Body Mass Index** | Excess adiposity exerts chronic hyper-physiologic joint compressive loading and systemic release of pro-inflammatory adipokines (leptin, adiponectin). |
        | `V00AGE` | **Patient Age** | Aging cartilage exhibits reduced proteoglycan content, decreased tensile strength, and impaired cellular repair mechanisms. |
        | `V00WOMKP` | **WOMAC Pain Score** | Ongoing nociceptive activation often corresponds with synovitis, bone marrow lesions (BMLs), and micro-instability. |
        | `V00INJ` | **Prior Joint Injury** | Prior ligamentous (ACL) or meniscal tears cause post-traumatic osteoarthritis (PTOA) via joint incongruence and focal loading. |
        | `V00PASE` | **Physical Activity** | Moderate cyclic loading maintains synovial fluid circulation, while extreme high-impact or sedentary status can negatively impact cartilage nutrition. |
        """)

    # =======================================================================
    # TAB 6: OAI Dataset Inspector
    # =======================================================================
    with tab_oai:
        st.markdown("### 🗂️ Osteoarthritis Initiative (OAI) Dataset Inspector")
        st.write("Real-time inspection and schema verification status of raw files in `data/raw/`.")

        if st.button("🔄 Scan & Refresh data/raw/ Directory"):
            from src.inspect_dataset import run_stage_1_inspection
            run_stage_1_inspection()
            st.success("Refreshed Stage 1 schema inspection.")

        schema_report = config.RESULTS_DIR / "oai_schema_report.md"
        if schema_report.exists():
            with open(schema_report, 'r', encoding='utf-8') as f:
                st.markdown(f.read())
        else:
            st.info("No schema report generated yet. Run `python -m src.inspect_dataset` to profile your files.")


if __name__ == "__main__":
    main()
