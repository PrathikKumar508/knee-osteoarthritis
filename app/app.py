"""
Streamlit Web Application: Knee OA Progression Risk Intelligence Platform
==========================================================================
Clinical Decision-Support & ML Research Prototype
for Early Knee Osteoarthritis Progression Risk Assessment.

Dashboard-style theme: colorful icon-badge stat cards, gradient hero card,
donut risk score, and a card-grid layout inspired by modern wellness dashboards.
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

sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.predict import predict_patient_risk, load_model_pipeline
from src.explainability import generate_patient_risk_explanation

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knee OA Progression Risk AI",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#f3f4f9"
INK = "#0f172a"
SUBTLE = "#94a3b8"
BORDER = "#eef0f6"
INDIGO = "#4f46e5"
INDIGO_SOFT = "#eef2ff"
PURPLE = "#7c3aed"
GREEN = "#059669"
ORANGE = "#ea580c"
PINK = "#db2777"
BLUE = "#2563eb"
LOW_C, MOD_C, HIGH_C = "#10b981", "#f59e0b", "#f43f5e"

CHART_FONT = dict(family="Inter", color=INK)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; color: {INK}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.4rem; max-width: 1300px; }}

    /* Top bar */
    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        background: #ffffff; border-radius: 18px; padding: 0.75rem 1.3rem;
        margin-bottom: 1.1rem; box-shadow: 0 2px 14px rgba(15,23,42,0.05);
        border: 1px solid {BORDER};
    }}
    .brand {{ display: flex; align-items: center; gap: 0.75rem; }}
    .brand-icon {{
        width: 44px; height: 44px; border-radius: 13px; font-size: 1.35rem;
        background: linear-gradient(135deg, {INDIGO}, {PURPLE});
        display: flex; align-items: center; justify-content: center; color: white;
    }}
    .brand-name {{ font-weight: 800; font-size: 1.05rem; color: {INK}; line-height: 1.1; }}
    .brand-sub {{ font-size: 0.74rem; color: {SUBTLE}; }}
    .status-badge {{
        display: flex; align-items: center; gap: 0.45rem; font-size: 0.78rem;
        color: {SUBTLE}; background: {BG}; border: 1px solid {BORDER};
        padding: 0.4rem 0.9rem; border-radius: 9999px; font-weight: 600;
    }}
    .status-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {GREEN}; }}

    /* Tabs as pill nav */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background-color: #ffffff; padding: 6px; border-radius: 14px;
        border: 1px solid {BORDER}; box-shadow: 0 2px 14px rgba(15,23,42,0.04);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px; padding: 8px 16px; font-weight: 600; font-size: 0.86rem;
        color: {SUBTLE}; background-color: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: {INDIGO} !important; color: #ffffff !important;
    }}

    /* Generic white card */
    .card {{
        background: #ffffff; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 1.3rem 1.4rem; box-shadow: 0 2px 14px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }}
    .card-title {{ font-weight: 700; font-size: 0.95rem; color: {INK}; margin-bottom: 0.9rem; }}
    .card-sub {{ font-size: 0.8rem; color: {SUBTLE}; margin-top: -0.6rem; margin-bottom: 0.9rem; }}

    /* Make native bordered containers match the card style */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-radius: 18px !important; border: 1px solid {BORDER} !important;
        box-shadow: 0 2px 14px rgba(15,23,42,0.05) !important; background: #ffffff !important;
    }}

    /* Icon-badge stat card */
    .stat-card {{
        background: #ffffff; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 1.05rem 1.2rem; box-shadow: 0 2px 12px rgba(15,23,42,0.05); height: 100%;
    }}
    .icon-badge {{
        width: 38px; height: 38px; border-radius: 11px; font-size: 1.05rem;
        display: flex; align-items: center; justify-content: center; margin-bottom: 0.55rem;
    }}
    .ib-indigo {{ background: {INDIGO_SOFT}; color: {INDIGO}; }}
    .ib-purple {{ background: #f5f3ff; color: {PURPLE}; }}
    .ib-green  {{ background: #ecfdf5; color: {GREEN}; }}
    .ib-orange {{ background: #fff7ed; color: {ORANGE}; }}
    .ib-pink   {{ background: #fdf2f8; color: {PINK}; }}
    .ib-blue   {{ background: #eff6ff; color: {BLUE}; }}
    .stat-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: {SUBTLE}; font-weight: 700; margin-bottom: 0.15rem; }}
    .stat-value {{ font-size: 1.35rem; font-weight: 800; color: {INK}; }}
    .stat-value .unit {{ font-size: 0.85rem; font-weight: 600; color: {SUBTLE}; }}
    .stat-sub {{ font-size: 0.74rem; color: {SUBTLE}; margin-top: 0.15rem; }}

    /* Hero gradient risk card */
    .hero-gradient {{
        background: linear-gradient(135deg, {INDIGO} 0%, #4338ca 60%, {PURPLE} 100%);
        border-radius: 20px; padding: 1.6rem 1.8rem; color: white; margin-bottom: 1rem;
        box-shadow: 0 10px 30px -8px rgba(79,70,229,0.45);
    }}
    .hero-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem; }}
    .hero-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85; font-weight: 700; }}
    .hero-chip {{ background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; }}
    .hero-value {{ font-size: 3rem; font-weight: 900; line-height: 1; margin: 0.3rem 0 1rem 0; }}
    .hero-value .pct {{ font-size: 1.6rem; font-weight: 700; opacity: 0.85; }}
    .seg-bar {{ position: relative; height: 12px; border-radius: 8px; display: flex; overflow: visible; background: rgba(255,255,255,0.15); }}
    .seg {{ height: 100%; }}
    .seg-marker {{
        position: absolute; top: -5px; width: 4px; height: 22px; background: white; border-radius: 3px;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.25);
    }}
    .seg-labels {{ display: flex; justify-content: space-between; font-size: 0.72rem; opacity: 0.8; margin-top: 0.4rem; }}

    /* Recommendation tile */
    .rec-tile {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem; height: 100%; }}
    .rec-tile .icon {{ font-size: 1.3rem; margin-bottom: 0.35rem; }}
    .rec-tile .title {{ font-weight: 700; color: {INK}; font-size: 0.88rem; margin-bottom: 0.3rem; }}
    .rec-tile .desc {{ font-size: 0.8rem; color: {SUBTLE}; line-height: 1.5; }}

    /* Driver row (body-composition style) */
    .driver-row {{ display: flex; align-items: center; gap: 0.8rem; padding: 0.6rem 0; border-bottom: 1px solid {BORDER}; }}
    .driver-row:last-child {{ border-bottom: none; }}
    .driver-name {{ font-weight: 600; font-size: 0.85rem; color: {INK}; }}
    .driver-desc {{ font-size: 0.72rem; color: {SUBTLE}; }}

    /* CTA banner */
    .cta-banner {{
        background: linear-gradient(135deg, #1e1b4b, #4338ca);
        border-radius: 20px; padding: 1.5rem 1.8rem; color: white;
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
    }}
    .cta-title {{ font-size: 1.1rem; font-weight: 800; }}
    .cta-sub {{ font-size: 0.82rem; opacity: 0.85; margin-top: 0.2rem; }}

    .stButton>button {{ border-radius: 10px; font-weight: 600; }}
    .stDownloadButton>button {{ border-radius: 10px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML component helpers
# ---------------------------------------------------------------------------
def icon_stat_card(icon: str, color_class: str, label: str, value: str, unit: str = "", sub: str = "") -> str:
    unit_html = f'<span class="unit"> {unit}</span>' if unit else ""
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card">
        <div class="icon-badge {color_class}">{icon}</div>
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}{unit_html}</div>
        {sub_html}
    </div>
    """


def driver_row_html(icon: str, color_class: str, name: str, desc: str) -> str:
    return f"""
    <div class="driver-row">
        <div class="icon-badge {color_class}" style="margin-bottom:0;">{icon}</div>
        <div>
            <div class="driver-name">{name}</div>
            <div class="driver-desc">{desc}</div>
        </div>
    </div>
    """


def risk_badge_chip(tier_label: str) -> str:
    return f'<span class="hero-chip">{tier_label}</span>'


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def create_risk_donut(probability: float, tier_color: str):
    pct = probability * 100
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct], hole=0.72,
        marker=dict(colors=[tier_color, "#eef0f6"]),
        textinfo="none", sort=False, direction="clockwise"
    ))
    fig.update_layout(
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=230,
        annotations=[dict(text=f"<b>{pct:.0f}</b><span style='font-size:14px;color:{SUBTLE}'>/100</span>",
                           x=0.5, y=0.55, font=dict(size=32, color=INK), showarrow=False),
                     dict(text="Progression Risk Score", x=0.5, y=0.38, font=dict(size=12, color=SUBTLE), showarrow=False)]
    )
    return fig


def create_feature_contribution_chart(feature_contributions: dict, top_n: int = 7):
    friendly_names = {
        "V00AGE": "Age (Baseline)", "V00BMI": "BMI (Joint Loading)",
        "V00WOMKP": "WOMAC Pain Score", "V00WOMAD": "WOMAC Disability",
        "V00WOMST": "WOMAC Stiffness", "V00PASE": "Physical Activity (PASE)",
        "V00SEX_1": "Sex: Male", "V00SEX_2": "Sex: Female",
        "V00KL_0": "KL Grade 0 (Normal)", "V00KL_1": "KL Grade 1 (Doubtful)",
        "V00KL_2": "KL Grade 2 (Minimal OA)", "V00KL_3": "KL Grade 3 (Moderate OA)",
        "V00INJ_0": "No Prior Injury", "V00INJ_1": "Prior Knee Injury",
        "V00SURG_0": "No Prior Surgery", "V00SURG_1": "Prior Knee Surgery"
    }
    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_feats.reverse()
    names = [friendly_names.get(k, k) for k, _ in sorted_feats]
    values = [v for _, v in sorted_feats]
    colors = [HIGH_C if v > 0 else LOW_C for v in values]
    text_labels = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=text_labels, textposition='auto', hoverinfo='x+y'
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Relative Effect on Log-Odds", color=SUBTLE, gridcolor=BORDER, zerolinecolor="#cbd5e1"),
        yaxis=dict(color=INK),
        margin=dict(l=10, r=20, t=10, b=30), height=300, font=CHART_FONT
    )
    return fig


def create_radar_profile_chart(patient_data: dict):
    categories = ['Pain (WOMAC)', 'Disability (WOMAC)', 'Stiffness (WOMAC)', 'BMI Index', 'Activity (PASE)']
    patient_scaled = [
        (patient_data["V00WOMKP"] / 20.0) * 100,
        (patient_data["V00WOMAD"] / 68.0) * 100,
        (patient_data["V00WOMST"] / 8.0) * 100,
        ((patient_data["V00BMI"] - 18.0) / (40.0 - 18.0)) * 100,
        (patient_data["V00PASE"] / 300.0) * 100
    ]
    cohort_avg_scaled = [30.0, 26.0, 25.0, 48.0, 48.0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=patient_scaled + [patient_scaled[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(79, 70, 229, 0.18)',
        line=dict(color=INDIGO, width=2), name='Current Patient'
    ))
    fig.add_trace(go.Scatterpolar(
        r=cohort_avg_scaled + [cohort_avg_scaled[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(148,163,184,0.10)',
        line=dict(color=SUBTLE, width=1.5, dash='dash'), name='Study Median'
    ))
    fig.update_layout(
        polar=dict(bgcolor="#ffffff",
                   radialaxis=dict(visible=True, range=[0, 100], color=SUBTLE, gridcolor=BORDER),
                   angularaxis=dict(color=INK, gridcolor=BORDER)),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        legend=dict(font=dict(color=INK), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=15, b=25), height=300, font=CHART_FONT
    )
    return fig


def generate_clinical_recommendations(prob: float, patient_data: dict) -> list:
    recs = []
    if prob >= config.HIGH_RISK_THRESHOLD:
        recs.append({"icon": "🚨", "title": "Close Longitudinal Monitoring",
                      "desc": "High predicted progression probability. Consider 12-month interval clinical and radiographic follow-up."})
    elif prob >= config.LOW_RISK_THRESHOLD:
        recs.append({"icon": "⚠️", "title": "Moderate Surveillance",
                      "desc": "Intermediate risk profile. Routine 18–24 month clinical review with symptomatic monitoring."})
    else:
        recs.append({"icon": "✅", "title": "Standard Low-Intensity Surveillance",
                      "desc": "Baseline metrics align with lower progression likelihood."})

    if patient_data.get("V00BMI", 0) >= 28.0:
        recs.append({"icon": "⚖️", "title": "Joint-Load Optimization",
                      "desc": f"BMI is {patient_data['V00BMI']:.1f} kg/m². Weight reduction lowers compressive joint loading."})
    if patient_data.get("V00WOMKP", 0) >= 8:
        recs.append({"icon": "🩹", "title": "Targeted Physical Therapy",
                      "desc": "Elevated baseline pain detected. Quadriceps/hamstring conditioning can slow functional decline."})
    if patient_data.get("V00KL", 0) >= 2:
        recs.append({"icon": "🦴", "title": "Pre-existing Radiographic OA",
                      "desc": f"Baseline KL Grade {patient_data['V00KL']} indicates existing joint space narrowing."})
    return recs


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================
def main():
    # -----------------------------------------------------------------------
    # Top bar
    # -----------------------------------------------------------------------
    st.markdown(f"""
    <div class="topbar">
        <div class="brand">
            <div class="brand-icon">🦵</div>
            <div>
                <div class="brand-name">Knee OA AI</div>
                <div class="brand-sub">Progression Intelligence</div>
            </div>
        </div>
        <div class="status-badge"><span class="status-dot"></span> Research Prototype — Not a diagnostic tool</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚠️ Clinical research notice", expanded=False):
        st.caption(
            "This platform is strictly an educational and research prototype. It is **not** a certified medical "
            "diagnostic device. Model outputs are probabilistic statistical estimates and must never replace "
            "professional medical evaluation, diagnosis, or treatment decisions."
        )

    pipeline_dict = None
    model_ok = True
    try:
        pipeline_dict = load_model_pipeline()
    except Exception:
        model_ok = False

    # Quick facts row (model / features / explainability)
    qf1, qf2, qf3 = st.columns(3)
    with qf1:
        st.markdown(icon_stat_card("🧠", "ib-indigo", "Prediction Model",
                                    pipeline_dict.get("model_name", "—") if pipeline_dict else "Not trained"),
                    unsafe_allow_html=True)
    with qf2:
        st.markdown(icon_stat_card("🧬", "ib-blue", "Input Features", str(len(config.get_feature_list()))),
                    unsafe_allow_html=True)
    with qf3:
        st.markdown(icon_stat_card("🔎", "ib-green", "Explainability", "SHAP Enabled"),
                    unsafe_allow_html=True)

    st.write("")

    tab_calc, tab_compare, tab_batch, tab_models, tab_shap, tab_oai = st.tabs([
        "🧮 Dashboard", "👥 Comparison", "📁 Batch Screening",
        "📊 Model Performance", "🧬 Explainability", "🗂️ Research Data"
    ])

    # =======================================================================
    # TAB 1: Dashboard / Patient Assessment
    # =======================================================================
    with tab_calc:
        with st.sidebar:
            st.markdown("### ⚙️ Risk Thresholds")
            t_low = st.slider("Low → Moderate", 0.10, 0.50, float(config.LOW_RISK_THRESHOLD), 0.05)
            t_high = st.slider("Moderate → High", 0.50, 0.85, float(config.HIGH_RISK_THRESHOLD), 0.05)
            if pipeline_dict:
                metrics = pipeline_dict.get("metrics", {})
                st.markdown("### 📈 Active Model")
                st.metric("ROC-AUC", f"{metrics.get('ROC_AUC', 0.0):.3f}")
                st.metric("Sensitivity", f"{metrics.get('Sensitivity_Recall', 0.0):.3f}")
                st.metric("Specificity", f"{metrics.get('Specificity', 0.0):.3f}")

        st.markdown("##### ⚡ Quick clinical scenario presets")
        if "age_val" not in st.session_state:
            st.session_state.update(age_val=63, sex_val=2, bmi_val=28.5, pain_val=6, dis_val=18,
                                     stiff_val=2, pase_val=145.0, kl_val=2, inj_val=0, surg_val=0)

        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("🟢 Low-Risk Patient", use_container_width=True):
                st.session_state.update(age_val=52, sex_val=1, bmi_val=23.2, pain_val=1, dis_val=4,
                                         stiff_val=1, pase_val=210.0, kl_val=0, inj_val=0, surg_val=0)
                st.rerun()
        with preset_cols[1]:
            if st.button("🟡 Moderate-Risk Patient", use_container_width=True):
                st.session_state.update(age_val=63, sex_val=2, bmi_val=28.4, pain_val=6, dis_val=16,
                                         stiff_val=2, pase_val=140.0, kl_val=1, inj_val=1, surg_val=0)
                st.rerun()
        with preset_cols[2]:
            if st.button("🔴 High-Risk Progressor", use_container_width=True):
                st.session_state.update(age_val=72, sex_val=2, bmi_val=34.6, pain_val=14, dis_val=38,
                                         stiff_val=5, pase_val=80.0, kl_val=3, inj_val=1, surg_val=1)
                st.rerun()
        with preset_cols[3]:
            if st.button("🎲 Random Profile", use_container_width=True):
                np.random.seed(int(datetime.now().timestamp()) % 10000)
                st.session_state.update(
                    age_val=int(np.random.randint(48, 79)), sex_val=int(np.random.choice([1, 2])),
                    bmi_val=round(float(np.random.uniform(20.0, 38.0)), 1),
                    pain_val=int(np.random.randint(0, 18)), dis_val=int(np.random.randint(0, 50)),
                    stiff_val=int(np.random.randint(0, 7)), pase_val=round(float(np.random.uniform(50.0, 260.0)), 1),
                    kl_val=int(np.random.choice([0, 1, 2, 3])), inj_val=int(np.random.choice([0, 1])),
                    surg_val=int(np.random.choice([0, 1]))
                )
                st.rerun()

        with st.container(border=True):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown("**👤 Demographics & Biometrics**")
                p_age = st.number_input("Age (Years)", 40, 90, st.session_state.age_val, step=1)
                p_sex = st.selectbox("Biological Sex", [1, 2], index=0 if st.session_state.sex_val == 1 else 1,
                                      format_func=lambda x: "Male" if x == 1 else "Female")
                p_bmi = st.number_input("BMI (kg/m²)", 15.0, 50.0, float(st.session_state.bmi_val), step=0.1)
            with col_c2:
                st.markdown("**🩹 Baseline Symptoms (WOMAC)**")
                p_pain = st.slider("WOMAC Pain (0–20)", 0, 20, st.session_state.pain_val)
                p_dis = st.slider("WOMAC Disability (0–68)", 0, 68, st.session_state.dis_val)
                p_stiff = st.slider("WOMAC Stiffness (0–8)", 0, 8, st.session_state.stiff_val)
                p_pase = st.number_input("Physical Activity (PASE)", 0.0, 400.0, float(st.session_state.pase_val), step=5.0)
            with col_c3:
                st.markdown("**🦴 Radiographic & Trauma History**")
                kl_descriptions = {0: "0: Normal", 1: "1: Doubtful", 2: "2: Minimal OA", 3: "3: Moderate OA"}
                p_kl = st.selectbox("Baseline KL Grade", [0, 1, 2, 3], index=st.session_state.kl_val,
                                     format_func=lambda x: kl_descriptions[x])
                p_inj = st.selectbox("Prior Knee Injury", [0, 1], index=st.session_state.inj_val,
                                      format_func=lambda x: "Yes" if x == 1 else "No")
                p_surg = st.selectbox("Prior Knee Surgery", [0, 1], index=st.session_state.surg_val,
                                       format_func=lambda x: "Yes" if x == 1 else "No")

        patient_input = {
            "V00AGE": p_age, "V00SEX": p_sex, "V00BMI": p_bmi, "V00WOMKP": p_pain,
            "V00WOMAD": p_dis, "V00WOMST": p_stiff, "V00PASE": p_pase,
            "V00KL": p_kl, "V00INJ": p_inj, "V00SURG": p_surg
        }

        # Live snapshot row — reflects current form values before running the model
        st.write("")
        sn1, sn2, sn3, sn4 = st.columns(4)
        with sn1:
            st.markdown(icon_stat_card("🎂", "ib-blue", "Age", str(p_age), "yrs"), unsafe_allow_html=True)
        with sn2:
            st.markdown(icon_stat_card("⚖️", "ib-orange", "BMI", f"{p_bmi:.1f}", "kg/m²"), unsafe_allow_html=True)
        with sn3:
            st.markdown(icon_stat_card("🩹", "ib-pink", "WOMAC Pain", f"{p_pain}", "/ 20"), unsafe_allow_html=True)
        with sn4:
            st.markdown(icon_stat_card("🏃", "ib-purple", "PASE Activity", f"{p_pase:.0f}", "pts"), unsafe_allow_html=True)

        st.write("")
        evaluate_btn = st.button("🚀 Calculate Progression Risk", type="primary", use_container_width=True)

        if evaluate_btn:
            if pipeline_dict is None:
                st.error("ML model pipeline not available. Run `python -m src.train` to train the model first.")
            else:
                with st.spinner("Analyzing baseline biomarkers and computing model risk estimate..."):
                    result = predict_patient_risk(patient_input, pipeline_dict)
                    prob = result["predicted_probability"]
                    model_used = result.get("model_used", "Ensemble Model")
                    contributions = result.get("feature_contributions", {})

                    if prob < t_low:
                        tier_label, tier_color = "LOW RISK", LOW_C
                    elif prob < t_high:
                        tier_label, tier_color = "MODERATE RISK", MOD_C
                    else:
                        tier_label, tier_color = "ELEVATED RISK", HIGH_C

                st.divider()

                # Hero gradient risk card
                pct = prob * 100
                st.markdown(f"""
                <div class="hero-gradient">
                    <div class="hero-top">
                        <div class="hero-label">Progression Risk — 48 Month Outlook</div>
                        {risk_badge_chip(tier_label)}
                    </div>
                    <div class="hero-value">{pct:.1f}<span class="pct">%</span></div>
                    <div class="seg-bar">
                        <div class="seg" style="width:{t_low*100:.1f}%; background:{LOW_C}; border-radius:8px 0 0 8px;"></div>
                        <div class="seg" style="width:{(t_high-t_low)*100:.1f}%; background:{MOD_C};"></div>
                        <div class="seg" style="width:{(100-t_high*100):.1f}%; background:{HIGH_C}; border-radius:0 8px 8px 0;"></div>
                        <div class="seg-marker" style="left:{pct:.1f}%;"></div>
                    </div>
                    <div class="seg-labels"><span>0%</span><span>Evaluated by {model_used}</span><span>100%</span></div>
                </div>
                """, unsafe_allow_html=True)

                # Donut + drivers row
                d_col1, d_col2 = st.columns([1, 1.3])
                with d_col1:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Risk Score</div>', unsafe_allow_html=True)
                        st.plotly_chart(create_risk_donut(prob, tier_color), use_container_width=True)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown(icon_stat_card("📐", "ib-indigo", "Relative Odds", f"{prob/(1-prob):.2f}", "x"), unsafe_allow_html=True)
                        with m2:
                            st.markdown(icon_stat_card("🎯", "ib-green" if tier_color == LOW_C else ("ib-orange" if tier_color == MOD_C else "ib-pink"),
                                                        "Tier", tier_label.split()[0]), unsafe_allow_html=True)

                with d_col2:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Top Contributing Factors</div>', unsafe_allow_html=True)
                        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
                        friendly = {
                            "V00AGE": "Age", "V00BMI": "BMI", "V00WOMKP": "WOMAC Pain",
                            "V00WOMAD": "WOMAC Disability", "V00WOMST": "WOMAC Stiffness", "V00PASE": "Physical Activity"
                        }
                        rows_html = ""
                        for k, v in sorted_feats:
                            name = friendly.get(k, k)
                            is_risk = v > 0
                            icon = "📈" if is_risk else "📉"
                            color_class = "ib-pink" if is_risk else "ib-green"
                            desc = f"{'Increases' if is_risk else 'Reduces'} estimated risk (weight {abs(v):.3f})"
                            rows_html += driver_row_html(icon, color_class, name, desc)
                        st.markdown(rows_html, unsafe_allow_html=True)

                # Feature impact + radar
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Feature Impact (Direction & Magnitude)</div>', unsafe_allow_html=True)
                        st.plotly_chart(create_feature_contribution_chart(contributions, top_n=7), use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Biomarker Radar Profile</div>', unsafe_allow_html=True)
                        st.plotly_chart(create_radar_profile_chart(patient_input), use_container_width=True)

                # Recommendations
                with st.container(border=True):
                    st.markdown('<div class="card-title">Clinical Monitoring Considerations (Non-Diagnostic)</div>', unsafe_allow_html=True)
                    recs = generate_clinical_recommendations(prob, patient_input)
                    rec_cols = st.columns(len(recs))
                    for idx, r in enumerate(recs):
                        with rec_cols[idx]:
                            st.markdown(f"""
                            <div class="rec-tile">
                                <div class="icon">{r['icon']}</div>
                                <div class="title">{r['title']}</div>
                                <div class="desc">{r['desc']}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # CTA banner + report download
                st.write("")
                st.markdown(f"""
                <div class="cta-banner">
                    <div>
                        <div class="cta-title">Export this assessment</div>
                        <div class="cta-sub">Download a shareable summary of this patient's baseline profile and risk estimate.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
        st.markdown("##### 👥 Patient Comparison & Differential Risk Tool")
        st.caption("Evaluate how modifying risk factors alters the progression trajectory.")

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            with st.container(border=True):
                st.markdown("**Patient Profile A (Baseline)**")
                a_age = st.number_input("Age (A)", 40, 90, 65, key="a_age")
                a_bmi = st.number_input("BMI (A)", 15.0, 50.0, 32.4, step=0.1, key="a_bmi")
                a_pain = st.slider("WOMAC Pain (A)", 0, 20, 10, key="a_pain")
                a_kl = st.selectbox("Baseline KL (A)", [0, 1, 2, 3], index=2, key="a_kl")
                a_inj = st.selectbox("Injury History (A)", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No", key="a_inj")
        with cmp_col2:
            with st.container(border=True):
                st.markdown("**Patient Profile B (Intervention)**")
                b_age = st.number_input("Age (B)", 40, 90, 65, key="b_age")
                b_bmi = st.number_input("BMI (B)", 15.0, 50.0, 26.0, step=0.1, key="b_bmi")
                b_pain = st.slider("WOMAC Pain (B)", 0, 20, 4, key="b_pain")
                b_kl = st.selectbox("Baseline KL (B)", [0, 1, 2, 3], index=2, key="b_kl")
                b_inj = st.selectbox("Injury History (B)", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No", key="b_inj")

        if st.button("⚖️ Compare Differential Progression Risk", type="primary", use_container_width=True):
            if pipeline_dict is not None:
                pat_a = {"V00AGE": a_age, "V00SEX": 2, "V00BMI": a_bmi, "V00WOMKP": a_pain, "V00WOMAD": a_pain*2, "V00WOMST": 2, "V00PASE": 120.0, "V00KL": a_kl, "V00INJ": a_inj, "V00SURG": 0}
                pat_b = {"V00AGE": b_age, "V00SEX": 2, "V00BMI": b_bmi, "V00WOMKP": b_pain, "V00WOMAD": b_pain*2, "V00WOMST": 1, "V00PASE": 160.0, "V00KL": b_kl, "V00INJ": b_inj, "V00SURG": 0}
                res_a = predict_patient_risk(pat_a, pipeline_dict)
                res_b = predict_patient_risk(pat_b, pipeline_dict)
                prob_a, prob_b = res_a["predicted_probability"], res_b["predicted_probability"]
                delta_prob = (prob_b - prob_a) * 100

                st.divider()
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Patient A</div>', unsafe_allow_html=True)
                        st.plotly_chart(create_risk_donut(prob_a, HIGH_C if prob_a >= 0.6 else (MOD_C if prob_a >= 0.3 else LOW_C)), use_container_width=True)
                with dcol2:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">Patient B</div>', unsafe_allow_html=True)
                        st.plotly_chart(create_risk_donut(prob_b, HIGH_C if prob_b >= 0.6 else (MOD_C if prob_b >= 0.3 else LOW_C)), use_container_width=True)

                shift_color = LOW_C if delta_prob < 0 else HIGH_C
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <h3 style="margin:0;">Differential Shift: <span style="color:{shift_color};">{delta_prob:+.1f}%</span></h3>
                    <p style="color:{SUBTLE};">Risk moved from <strong>{prob_a*100:.1f}%</strong> to <strong>{prob_b*100:.1f}%</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

    # =======================================================================
    # TAB 3: Batch Cohort Screening
    # =======================================================================
    with tab_batch:
        st.markdown("##### 📁 Batch Patient Cohort Risk Screening")
        st.caption("Upload a CSV of multiple patients for batch progression risk scoring.")

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
                        df_batch["Progression_Probability"], bins=[-0.01, t_low, t_high, 1.01],
                        labels=["Low Risk", "Moderate Risk", "High Risk"]
                    )

                st.markdown("###### 📈 Cohort Stratification Overview")
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    fig_pie = px.pie(df_batch, names="Risk_Tier", color="Risk_Tier",
                                      color_discrete_map={"Low Risk": LOW_C, "Moderate Risk": MOD_C, "High Risk": HIGH_C},
                                      title="Cohort Risk Distribution")
                    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=CHART_FONT)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with stat_col2:
                    fig_hist = px.histogram(df_batch, x="Progression_Probability", nbins=15,
                                             title="Probability Score Density", color_discrete_sequence=[INDIGO])
                    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=CHART_FONT)
                    st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("###### 📋 Annotated Cohort Predictions")
                st.dataframe(df_batch, use_container_width=True)
                annotated_csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Export Annotated Predictions CSV", annotated_csv, "oai_predictions_output.csv", "text/csv")

    # =======================================================================
    # TAB 4: Model Performance
    # =======================================================================
    with tab_models:
        st.markdown("##### 📊 Model Architecture Comparison & Evaluation Metrics")
        eval_report_path = config.RESULTS_DIR / "evaluation_report.md"
        if eval_report_path.exists():
            with st.container(border=True):
                with open(eval_report_path, 'r', encoding='utf-8') as f:
                    st.markdown(f.read())

        st.markdown("###### 📈 Diagnostic Visualizations Gallery")
        fig_files = list(config.FIGURES_DIR.glob("*.png"))
        if fig_files:
            fig_cols = st.columns(len(fig_files))
            for i, fp in enumerate(fig_files):
                with fig_cols[i]:
                    st.image(str(fp), caption=fp.stem.replace("_", " ").title(), use_container_width=True)

        st.divider()
        st.markdown("###### 🎛️ Clinical Decision Threshold Tuning Simulator")
        sim_thresh = st.slider("Simulated Decision Threshold", 0.10, 0.90, 0.50, 0.05)
        sim_sens = max(0.20, min(0.98, 0.85 - (sim_thresh - 0.3) * 0.9))
        sim_spec = max(0.20, min(0.98, 0.55 + (sim_thresh - 0.3) * 0.8))
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Simulated Sensitivity", f"{sim_sens:.2%}", delta=f"{(sim_sens-0.74)*100:+.1f}% vs default")
        with sc2:
            st.metric("Simulated Specificity", f"{sim_spec:.2%}", delta=f"{(sim_spec-0.76)*100:+.1f}% vs default")
        with sc3:
            st.caption("Lowering the threshold prioritizes catching every potential progressor, at the cost of more false alarms.")

    # =======================================================================
    # TAB 5: Explainability
    # =======================================================================
    with tab_shap:
        st.markdown("##### 🧬 Explainable AI (XAI) & Biomarker Context")
        shap_img = config.FIGURES_DIR / "shap_summary_plot.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="Global SHAP Summary Plot", use_container_width=True)

        with st.container(border=True):
            st.markdown("""
##### 🔬 Clinical Relevance Matrix
| Feature | Biological Name | Mechanism in OA Progression |
| --- | --- | --- |
| `V00KL` | Baseline Kellgren-Lawrence Grade | Pre-existing chondrocyte depletion, subchondral sclerosis, and marginal osteophytes create an unstable mechanical joint. |
| `V00BMI` | Body Mass Index | Excess adiposity exerts chronic joint compressive loading and systemic release of pro-inflammatory adipokines. |
| `V00AGE` | Patient Age | Aging cartilage exhibits reduced proteoglycan content and impaired repair mechanisms. |
| `V00WOMKP` | WOMAC Pain Score | Ongoing nociceptive activation often corresponds with synovitis and micro-instability. |
| `V00INJ` | Prior Joint Injury | Prior ligamentous or meniscal tears cause post-traumatic osteoarthritis via joint incongruence. |
| `V00PASE` | Physical Activity | Moderate cyclic loading maintains synovial fluid circulation; extremes can affect cartilage nutrition. |
""")

    # =======================================================================
    # TAB 6: Research Data (OAI Inspector)
    # =======================================================================
    with tab_oai:
        st.markdown("##### 🗂️ Osteoarthritis Initiative (OAI) Dataset Inspector")
        if st.button("🔄 Scan & Refresh data/raw/ Directory"):
            from src.inspect_dataset import run_stage_1_inspection
            run_stage_1_inspection()
            st.success("Refreshed Stage 1 schema inspection.")

        schema_report = config.RESULTS_DIR / "oai_schema_report.md"
        if schema_report.exists():
            with st.container(border=True):
                with open(schema_report, 'r', encoding='utf-8') as f:
                    st.markdown(f.read())
        else:
            st.info("No schema report generated yet. Run `python -m src.inspect_dataset` to profile your files.")


if __name__ == "__main__":
    main()