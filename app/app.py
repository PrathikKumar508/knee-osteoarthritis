"""
Streamlit Web Application: Knee OA Progression Risk
====================================================
Clinical decision-support research prototype for early knee osteoarthritis
progression risk assessment.

Design: clinical instrument. Cool neutral surface, one petrol accent, hairline
rules, small radii, tabular numerals. The risk scale strip is the single
featured element; everything around it is deliberately quiet.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from src.config import config
from src.predict import predict_patient_risk, load_model_pipeline, list_available_models

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knee OA Progression Risk",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#f3f5f7"
SURFACE = "#ffffff"
INK = "#14202b"
MUTED = "#566473"
FAINT = "#8895a3"
RULE = "#d5dbe1"
ACCENT = "#0f5c70"

LOW_C, MOD_C, HIGH_C = "#2f7d5b", "#b7791f", "#b03a48"
LOW_T, MOD_T, HIGH_T = "#dcebe3", "#f3e6c8", "#f0d5d8"

SANS = "'IBM Plex Sans', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"
CHART_FONT = dict(family="IBM Plex Sans, sans-serif", color=INK, size=12)

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');"
)

ROOT_VARS = (
    ":root{"
    f"--bg:{BG};--surface:{SURFACE};--ink:{INK};--muted:{MUTED};--faint:{FAINT};"
    f"--rule:{RULE};--accent:{ACCENT};"
    f"--low:{LOW_C};--mod:{MOD_C};--high:{HIGH_C};"
    f"--low-t:{LOW_T};--mod-t:{MOD_T};--high-t:{HIGH_T};"
    f"--sans:{SANS};--mono:{MONO};"
    "}"
)

CSS = """
html, body, .stApp, .stMarkdown, .stMarkdown p, label, input, textarea, button,
[data-baseweb="tab"], [data-testid="stCaptionContainer"] { font-family: var(--sans); }
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: "Material Symbols Rounded" !important;
}
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }

/* Header */
.hdr { display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem;
       flex-wrap: wrap; padding-bottom: 0.9rem; border-bottom: 2px solid var(--ink); }
.hdr-title { font-size: 1.45rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.15; }
.hdr-sub { font-size: 0.86rem; color: var(--muted); margin-top: 0.25rem; }
.hdr-note { display: flex; align-items: center; font-size: 0.8rem; color: var(--muted); }
.hdr-note .sep { width: 1px; height: 14px; background: var(--rule); margin: 0 0.75rem; }
.dot { width: 8px; height: 8px; border-radius: 1px; display: inline-block; margin-right: 0.5rem; }
.dot.on { background: var(--low); }
.dot.off { background: var(--high); }

/* Meta strip */
.meta { display: flex; flex-wrap: wrap; background: var(--surface); border: 1px solid var(--rule);
        border-top: none; margin-bottom: 1.2rem; }
.meta-cell { flex: 1 1 180px; padding: 0.65rem 1rem; border-right: 1px solid var(--rule); }
.meta-cell:last-child { border-right: none; }
.meta-k { font-size: 0.74rem; color: var(--faint); }
.meta-v { font-family: var(--mono); font-size: 0.92rem; color: var(--ink); margin-top: 0.1rem; }

/* Section headings */
.sec { margin: 1.9rem 0 0.8rem 0; padding-bottom: 0.45rem; border-bottom: 1px solid var(--rule); }
.sec-title { font-size: 1.02rem; font-weight: 600; }
.sec-sub { font-size: 0.82rem; color: var(--muted); margin-top: 0.15rem; }
.grp { font-size: 0.86rem; font-weight: 600; color: var(--ink); margin: 0.1rem 0 0.5rem 0;
       padding-bottom: 0.3rem; border-bottom: 1px solid var(--rule); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; border-bottom: 1px solid var(--rule); }
.stTabs [data-baseweb="tab"] { padding: 0.6rem 1.15rem; font-size: 0.86rem; font-weight: 500;
                               color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }

/* Bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 3px !important; border: 1px solid var(--rule) !important;
    box-shadow: none !important; background: var(--surface) !important;
}

/* Inputs and buttons */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { border-radius: 2px !important; }
.stButton > button, .stDownloadButton > button {
    border-radius: 2px; border: 1px solid var(--ink); background: var(--surface);
    color: var(--ink); font-weight: 500; font-size: 0.85rem; box-shadow: none;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent); color: var(--accent); background: var(--surface);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent); border-color: var(--accent); color: #ffffff;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #0b4657; border-color: #0b4657; color: #ffffff;
}
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--rule); }
[data-testid="stMetricValue"] { font-family: var(--mono); font-size: 1.3rem; }

/* Readout panel */
.readout { display: flex; flex-wrap: wrap; background: var(--surface); border: 1px solid var(--rule);
           border-radius: 3px; margin-bottom: 0.9rem; }
.ro-left { flex: 0 0 250px; padding: 1.15rem 1.4rem; border-right: 1px solid var(--rule); }
.ro-label { font-size: 0.82rem; color: var(--muted); }
.ro-num { font-family: var(--mono); font-size: 3.6rem; font-weight: 500; line-height: 1.05;
          letter-spacing: -0.03em; margin: 0.25rem 0 0.6rem 0; }
.ro-num span { font-size: 1.6rem; color: var(--muted); margin-left: 0.15rem; }
.ro-tier { display: inline-block; border-left: 4px solid; padding-left: 0.6rem;
           font-size: 0.86rem; font-weight: 600; }
.ro-right { flex: 1 1 320px; padding: 0.9rem 2rem 1rem 2rem; display: flex; flex-direction: column; justify-content: center; }
.ro-note { font-size: 0.8rem; color: var(--muted); margin-top: 0.9rem; }

/* Risk scale strip */
.scale { position: relative; padding-top: 34px; }
.scale-track { position: relative; height: 24px; background: var(--rule); border-radius: 2px; }
.scale-seg { position: absolute; top: 0; bottom: 0; font-size: 0.72rem; line-height: 24px;
             text-align: center; color: var(--muted); overflow: hidden; white-space: nowrap; }
.scale-seg.low { background: var(--low-t); border-radius: 2px 0 0 2px; }
.scale-seg.mod { background: var(--mod-t); }
.scale-seg.high { background: var(--high-t); border-radius: 0 2px 2px 0; }
.scale-marker { position: absolute; top: -9px; bottom: -9px; width: 2px; background: var(--ink); transform: translateX(-1px); }
.scale-marker .lbl { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
                     font-family: var(--mono); font-size: 0.82rem; font-weight: 500;
                     padding-bottom: 3px; white-space: nowrap; color: var(--ink); }
.scale-marker.scn { background: var(--accent); }
.scale-marker.scn .lbl { bottom: auto; top: 100%; padding: 3px 0 0 0; color: var(--accent); }
.scale-ticks { position: relative; height: 16px; margin-top: 12px; }
.scale.has-scn .scale-ticks { margin-top: 30px; }
.scale-ticks .tick { position: absolute; transform: translateX(-50%); font-family: var(--mono);
                     font-size: 0.72rem; color: var(--faint); }
.scale-ticks .tick::before { content: ""; position: absolute; left: 50%; top: -8px; height: 5px;
                             border-left: 1px solid var(--faint); }

/* Driver rows */
.drv { display: grid; grid-template-columns: 1fr auto; column-gap: 1rem; padding: 0.55rem 0;
       border-bottom: 1px solid var(--rule); }
.drv:last-child { border-bottom: none; }
.drv-name { font-size: 0.88rem; font-weight: 500; }
.drv-dir { font-size: 0.76rem; color: var(--muted); }
.drv-val { font-family: var(--mono); font-size: 0.9rem; align-self: center; grid-row: 1 / span 2; grid-column: 2; }
.drv-val.up { color: var(--high); }
.drv-val.down { color: var(--low); }
.drv-bar { grid-column: 1 / span 2; height: 3px; background: var(--bg); margin-top: 0.4rem; }
.drv-bar span { display: block; height: 100%; }
.drv-bar span.up { background: var(--high); }
.drv-bar span.down { background: var(--low); }
.empty { font-size: 0.84rem; color: var(--muted); padding: 0.5rem 0; }

/* Monitoring notes */
.rec { border-left: 3px solid var(--accent); background: var(--surface); border-top: 1px solid var(--rule);
       border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
       padding: 0.75rem 1rem; margin-bottom: 0.5rem; }
.rec-title { font-size: 0.9rem; font-weight: 600; }
.rec-desc { font-size: 0.84rem; color: var(--muted); margin-top: 0.15rem; line-height: 1.5; }

/* Scenario delta */
.delta { font-size: 0.9rem; margin-top: 0.6rem; }
.delta b { font-family: var(--mono); font-weight: 500; }
.delta .good { color: var(--low); }
.delta .bad { color: var(--high); }
.delta .flat { color: var(--muted); }
.stale { font-size: 0.82rem; color: var(--mod); border-left: 3px solid var(--mod);
         padding: 0.35rem 0.7rem; background: var(--surface); margin: 0.6rem 0; }
"""

st.markdown(f"<style>{FONT_IMPORT}{ROOT_VARS}{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def render(html_str: str) -> None:
    """Collapse whitespace so markdown never treats indented HTML as a code block."""
    st.markdown(" ".join(html_str.split()), unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    render(f'<div class="sec"><div class="sec-title">{title}</div>{sub_html}</div>')


def group(label: str) -> None:
    render(f'<div class="grp">{label}</div>')


def meta_strip(items) -> None:
    cells = "".join(
        f'<div class="meta-cell"><div class="meta-k">{k}</div><div class="meta-v">{v}</div></div>'
        for k, v in items
    )
    render(f'<div class="meta">{cells}</div>')


def tier_of(prob: float, t_low: float, t_high: float):
    if prob < t_low:
        return "LOW RISK", LOW_C
    if prob < t_high:
        return "MODERATE RISK", MOD_C
    return "ELEVATED RISK", HIGH_C


def risk_scale_html(prob: float, t_low: float, t_high: float, scenario: float = None) -> str:
    lo, hi = t_low * 100, t_high * 100
    pct = prob * 100
    scn_cls = " has-scn" if scenario is not None else ""
    scn_html = ""
    if scenario is not None:
        scn_html = (
            f'<div class="scale-marker scn" style="left:{scenario * 100:.2f}%;">'
            f'<span class="lbl">{scenario * 100:.1f}%</span></div>'
        )
    ticks = "".join(
        f'<span class="tick" style="left:{p:.2f}%;">{p:.0f}%</span>' for p in (0, lo, hi, 100)
    )
    return (
        f'<div class="scale{scn_cls}"><div class="scale-track">'
        f'<div class="scale-seg low" style="left:0;width:{lo:.2f}%;">Low</div>'
        f'<div class="scale-seg mod" style="left:{lo:.2f}%;width:{hi - lo:.2f}%;">Moderate</div>'
        f'<div class="scale-seg high" style="left:{hi:.2f}%;width:{100 - hi:.2f}%;">Elevated</div>'
        f'<div class="scale-marker" style="left:{pct:.2f}%;"><span class="lbl">{pct:.1f}%</span></div>'
        f"{scn_html}</div>"
        f'<div class="scale-ticks">{ticks}</div></div>'
    )


def readout_html(label: str, prob: float, t_low: float, t_high: float, note: str = "", scenario: float = None) -> str:
    tier_label, tier_color = tier_of(prob, t_low, t_high)
    return (
        '<div class="readout">'
        f'<div class="ro-left"><div class="ro-label">{label}</div>'
        f'<div class="ro-num">{prob * 100:.1f}<span>%</span></div>'
        f'<div class="ro-tier" style="border-color:{tier_color};color:{tier_color};">{tier_label}</div></div>'
        f'<div class="ro-right">{risk_scale_html(prob, t_low, t_high, scenario)}'
        f'<div class="ro-note">{note}</div></div></div>'
    )


# ---------------------------------------------------------------------------
# Feature naming and grouping
# ---------------------------------------------------------------------------
# Matched by prefix so it survives whatever suffix the one-hot encoder adds
# (e.g. "V00KL_0", "V00KL_0.0", "V00KL_2.0").
FRIENDLY_BASE_NAMES = {
    "V00AGE": "Age", "V00BMI": "BMI",
    "V00WOMKP": "WOMAC pain", "V00WOMAD": "WOMAC disability",
    "V00WOMST": "WOMAC stiffness", "V00PASE": "Physical activity (PASE)",
}
FRIENDLY_CATEGORY_LABELS = {
    "V00SEX": {"1": "Sex: male", "2": "Sex: female"},
    "V00KL": {"0": "KL grade 0 (normal)", "1": "KL grade 1 (doubtful)",
              "2": "KL grade 2 (minimal OA)", "3": "KL grade 3 (moderate OA)"},
    "V00INJ": {"0": "No prior injury", "1": "Prior knee injury"},
    "V00SURG": {"0": "No prior surgery", "1": "Prior knee surgery"},
}
ALL_BASES = ["V00AGE", "V00SEX", "V00BMI", "V00WOMKP", "V00WOMAD", "V00WOMST",
             "V00PASE", "V00KL", "V00INJ", "V00SURG"]
MODIFIABLE = {"V00BMI", "V00WOMKP", "V00WOMAD", "V00WOMST", "V00PASE"}


def friendly_feature_name(fname: str) -> str:
    if fname in FRIENDLY_BASE_NAMES:
        return FRIENDLY_BASE_NAMES[fname]
    for base, labels in FRIENDLY_CATEGORY_LABELS.items():
        if fname.startswith(base + "_"):
            code = fname[len(base) + 1:].split(".")[0]
            return labels.get(code, fname)
    return fname


def base_name(fname: str) -> str:
    for base in ALL_BASES:
        if fname == base or fname.startswith(base + "_"):
            return base
    return fname


def driver_rows_html(items) -> str:
    """Bars are scaled against THIS group's own largest contribution, so a
    group whose biggest factor is small doesn't get visually flattened by a
    shared scale borrowed from a different, larger group."""
    if not items:
        return '<div class="empty">No factors in this group.</div>'
    vmax = max((abs(v) for _, v in items), default=0.0)
    rows = ""
    for k, v in items:
        cls = "up" if v > 0 else "down"
        width = min(100.0, abs(v) / vmax * 100) if vmax else 0
        rows += (
            f'<div class="drv"><div class="drv-name">{friendly_feature_name(k)}</div>'
            f'<div class="drv-dir">{"Raises" if v > 0 else "Lowers"} estimated risk</div>'
            f'<div class="drv-val {cls}">{v:+.3f}</div>'
            f'<div class="drv-bar"><span class="{cls}" style="width:{width:.0f}%;"></span></div></div>'
        )
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style_fig(fig, height: int = 300, margin: dict = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, font=CHART_FONT,
        margin=margin or dict(l=10, r=20, t=10, b=30),
    )
    fig.update_xaxes(gridcolor=RULE, zerolinecolor=FAINT, linecolor=RULE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=RULE, linecolor=RULE, tickfont=dict(color=INK))
    return fig


SCALE_AXIS_TITLE = {
    "probability": "Effect on predicted probability (positive raises risk)",
    "log_odds": "Effect on log-odds (positive raises risk)",
    "relative_importance": "Relative importance (not directional; older model artifact)",
}
SCALE_SHORT_LABEL = {
    "probability": "probability scale",
    "log_odds": "log-odds scale",
    "relative_importance": "relative importance, not a per-patient effect size",
}


def create_feature_contribution_chart(feature_contributions: dict, scale: str = "log_odds", top_n: int = 8):
    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    sorted_feats.reverse()
    names = [friendly_feature_name(k) for k, _ in sorted_feats]
    values = [v for _, v in sorted_feats]
    colors = [HIGH_C if v > 0 else LOW_C for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in values], textposition="outside", cliponaxis=False,
        hoverinfo="x+y",
    ))
    fig.update_xaxes(title=SCALE_AXIS_TITLE.get(scale, SCALE_AXIS_TITLE["log_odds"]))
    return style_fig(fig, height=320, margin=dict(l=10, r=50, t=10, b=40))


def create_profile_chart(pdata: dict):
    """Patient position within each reference range versus the study median."""
    def clip(x):
        return float(max(0.0, min(100.0, x)))

    rows = [
        ("WOMAC pain", clip(pdata["V00WOMKP"] / 20.0 * 100), 30.0, f"{pdata['V00WOMKP']} of 20"),
        ("WOMAC disability", clip(pdata["V00WOMAD"] / 68.0 * 100), 26.0, f"{pdata['V00WOMAD']} of 68"),
        ("WOMAC stiffness", clip(pdata["V00WOMST"] / 8.0 * 100), 25.0, f"{pdata['V00WOMST']} of 8"),
        ("BMI", clip((pdata["V00BMI"] - 18.0) / (40.0 - 18.0) * 100), 48.0, f"{pdata['V00BMI']:.1f} kg/m2"),
        ("Physical activity", clip(pdata["V00PASE"] / 300.0 * 100), 48.0, f"PASE {pdata['V00PASE']:.0f}"),
    ]
    fig = go.Figure()
    for name, pat, med, _ in rows:
        fig.add_trace(go.Scatter(x=[med, pat], y=[name, name], mode="lines",
                                 line=dict(color=RULE, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[r[2] for r in rows], y=[r[0] for r in rows], mode="markers", name="Study median",
        marker=dict(symbol="diamond-open", size=10, color=MUTED, line=dict(width=1.5)),
        hovertemplate="%{y}: study median<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[r[1] for r in rows], y=[r[0] for r in rows], mode="markers", name="This patient",
        marker=dict(symbol="square", size=10, color=ACCENT),
        text=[r[3] for r in rows], hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_xaxes(range=[-3, 103], tickvals=[0, 25, 50, 75, 100],
                     title="Position within reference range (%)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                  font=dict(color=INK)))
    return style_fig(fig, height=320, margin=dict(l=10, r=20, t=30, b=40))


def generate_clinical_recommendations(prob: float, patient_data: dict) -> list:
    recs = []
    if prob >= config.HIGH_RISK_THRESHOLD:
        recs.append({"title": "Close longitudinal monitoring",
                     "desc": "High predicted progression probability. Consider 12-month interval clinical and radiographic follow-up."})
    elif prob >= config.LOW_RISK_THRESHOLD:
        recs.append({"title": "Moderate surveillance",
                     "desc": "Intermediate risk profile. Routine 18 to 24 month clinical review with symptomatic monitoring."})
    else:
        recs.append({"title": "Standard low-intensity surveillance",
                     "desc": "Baseline metrics align with lower progression likelihood."})

    if patient_data.get("V00BMI", 0) >= 28.0:
        recs.append({"title": "Joint-load optimization",
                     "desc": f"BMI is {patient_data['V00BMI']:.1f} kg/m2. Weight reduction lowers compressive joint loading."})
    if patient_data.get("V00WOMKP", 0) >= 8:
        recs.append({"title": "Targeted physical therapy",
                     "desc": "Elevated baseline pain detected. Quadriceps and hamstring conditioning can slow functional decline."})
    if patient_data.get("V00KL", 0) >= 2:
        recs.append({"title": "Pre-existing radiographic OA",
                     "desc": f"Baseline KL grade {patient_data['V00KL']} indicates existing joint space narrowing."})
    return recs


# ---------------------------------------------------------------------------
# Assessment result rendering
# ---------------------------------------------------------------------------
def render_scenario_panel(assess: dict, pipeline_dict, t_low: float, t_high: float) -> None:
    base = assess["input"]
    prob0 = assess["prob"]
    n = assess["n"]

    section("Scenario analysis",
            "Adjust modifiable measures to see how the estimate moves. Fixed factors stay at their assessed values.")
    col_in, col_out = st.columns([1, 1.5])
    with col_in:
        with st.container(border=True):
            s_bmi = st.slider("BMI (kg/m2)", 15.0, 50.0, float(base["V00BMI"]), 0.1, key=f"wi_bmi_{n}")
            s_pase = st.slider("Physical activity (PASE)", 0.0, 400.0, float(base["V00PASE"]), 5.0, key=f"wi_pase_{n}")
            s_pain = st.slider("WOMAC pain (0 to 20)", 0, 20, int(base["V00WOMKP"]), key=f"wi_pain_{n}")
            s_dis = st.slider("WOMAC disability (0 to 68)", 0, 68, int(base["V00WOMAD"]), key=f"wi_dis_{n}")
            s_stiff = st.slider("WOMAC stiffness (0 to 8)", 0, 8, int(base["V00WOMST"]), key=f"wi_stiff_{n}")

    scenario = dict(base, V00BMI=s_bmi, V00PASE=s_pase, V00WOMKP=s_pain, V00WOMAD=s_dis, V00WOMST=s_stiff)
    prob_s = predict_patient_risk(scenario, pipeline_dict, model_name=assess.get("model_name"))["predicted_probability"]
    delta = (prob_s - prob0) * 100
    tier0, _ = tier_of(prob0, t_low, t_high)
    tier_s, _ = tier_of(prob_s, t_low, t_high)

    if abs(delta) < 0.05:
        cls, verb = "flat", "No change"
    elif delta < 0:
        cls, verb = "good", "Lower"
    else:
        cls, verb = "bad", "Higher"
    tier_txt = f" Tier stays {tier0.lower()}." if tier0 == tier_s else f" Tier moves from {tier0.lower()} to {tier_s.lower()}."

    with col_out:
        render(
            '<div class="readout">'
            f'<div class="ro-right" style="padding:0.9rem 2rem 1rem 2rem;">'
            f"{risk_scale_html(prob0, t_low, t_high, scenario=prob_s)}"
            f'<div class="delta"><span class="{cls}">{verb}: <b>{delta:+.1f}</b> percentage points.</span>{tier_txt}</div>'
            '<div class="ro-note">Black marker: assessed profile. Petrol marker: scenario. '
            "The model reports statistical association, not the causal effect of an intervention.</div>"
            "</div></div>"
        )


def render_assessment(assess: dict, pipeline_dict, t_low: float, t_high: float, current_input: dict) -> None:
    inp = assess["input"]
    prob = assess["prob"]
    model_used = assess["model_used"]
    contributions = assess["contributions"]
    tier_label, _ = tier_of(prob, t_low, t_high)

    section("Result", "Estimated probability of radiographic progression at 48 months.")
    if current_input != inp:
        render('<div class="stale">Inputs have changed since this result was calculated. Recalculate to update.</div>')

    odds = prob / max(1 - prob, 1e-9)
    note = (
        f"Estimated by {model_used}. Odds {odds:.2f} to 1. "
        f"Tier thresholds: low below {t_low:.0%}, elevated from {t_high:.0%}."
    )
    render(readout_html("Progression probability", prob, t_low, t_high, note))

    # Contributing factors, split by whether they can be changed
    scale = assess.get("scale", "log_odds")
    section("Contributing factors",
            f"Largest model contributions, grouped by whether the measure can be changed. "
            f"Values are on the {SCALE_SHORT_LABEL.get(scale, scale)}.")
    if contributions:
        ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        fixed = [kv for kv in ranked if base_name(kv[0]) not in MODIFIABLE][:4]
        modif = [kv for kv in ranked if base_name(kv[0]) in MODIFIABLE][:4]
        f_col, m_col = st.columns(2)
        with f_col:
            with st.container(border=True):
                group("Fixed at baseline")
                render(driver_rows_html(fixed))
        with m_col:
            with st.container(border=True):
                group("Modifiable")
                render(driver_rows_html(modif))

        c1, c2 = st.columns([1.2, 1])
        with c1:
            with st.container(border=True):
                group("All factors, direction and magnitude")
                st.plotly_chart(create_feature_contribution_chart(contributions, scale=scale), use_container_width=True)
        with c2:
            with st.container(border=True):
                group("Profile against study median")
                st.plotly_chart(create_profile_chart(inp), use_container_width=True)
    else:
        st.caption("Feature contributions are not available for the active model.")

    render_scenario_panel(assess, pipeline_dict, t_low, t_high)

    # Monitoring notes
    section("Monitoring considerations", "Non-diagnostic prompts derived from the estimate and baseline measures.")
    recs_html = "".join(
        f'<div class="rec"><div class="rec-title">{r["title"]}</div><div class="rec-desc">{r["desc"]}</div></div>'
        for r in generate_clinical_recommendations(prob, inp)
    )
    render(recs_html)

    # Export
    section("Export", "Download a plain-text summary of this assessment.")
    report_text = f"""# KNEE OA PROGRESSION RISK ASSESSMENT REPORT
Date: {assess['ts'].strftime('%Y-%m-%d %H:%M:%S')}
Model Architecture: {model_used}
Target Window: 48-Month Follow-Up (Radiographic Progression Delta KL >= 1)

---
PATIENT BASELINE PARAMETERS:
- Age: {inp['V00AGE']} years
- Biological Sex: {'Male' if inp['V00SEX'] == 1 else 'Female'}
- BMI: {inp['V00BMI']:.1f} kg/m2
- Baseline Kellgren-Lawrence Grade: {inp['V00KL']}
- WOMAC Pain Score: {inp['V00WOMKP']}/20
- WOMAC Disability: {inp['V00WOMAD']}/68
- WOMAC Stiffness: {inp['V00WOMST']}/8
- Physical Activity (PASE): {inp['V00PASE']}
- History of Knee Injury: {'Yes' if inp['V00INJ'] == 1 else 'No'}
- History of Knee Surgery: {'Yes' if inp['V00SURG'] == 1 else 'No'}

---
PROGNOSTIC ML RESULTS:
- Predicted Progression Probability: {prob * 100:.2f}%
- Risk Stratification Tier: {tier_label}
- Odds: {odds:.3f}

KEY CONTRIBUTING FACTORS:
{chr(10).join([f"- {k}: {v:+.4f}" for k, v in list(contributions.items())[:6]])}

---
DISCLAIMER:
This report is generated by a research machine learning prototype and does NOT constitute a clinical diagnosis.
"""
    st.download_button(
        label="Download assessment summary (.txt)",
        data=report_text,
        file_name=f"OA_Risk_Report_{assess['ts'].strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================
def main():
    pipeline_dict = None
    try:
        pipeline_dict = load_model_pipeline()
    except Exception:
        pipeline_dict = None
    model_ok = pipeline_dict is not None
    metrics = pipeline_dict.get("metrics", {}) if pipeline_dict else {}

    # Header
    render(
        '<div class="hdr"><div>'
        '<div class="hdr-title">Knee OA progression risk</div>'
        '<div class="hdr-sub">Estimated probability of radiographic progression at 48 months, from baseline measures</div>'
        "</div>"
        '<div class="hdr-note">'
        f'<span class="dot {"on" if model_ok else "off"}"></span>{"Model loaded" if model_ok else "Model not trained"}'
        '<span class="sep"></span>Research prototype, not a diagnostic device'
        "</div></div>"
    )
    meta_strip([
        ("Prediction model", pipeline_dict.get("model_name", "Unknown") if model_ok else "Not trained"),
        ("Input features", str(len(config.get_feature_list()))),
        ("Explainability", "SHAP"),
        ("ROC-AUC", f"{metrics.get('ROC_AUC', 0.0):.3f}" if model_ok else "n/a"),
    ])

    with st.expander("Research notice", expanded=False):
        st.caption(
            "This platform is strictly an educational and research prototype. It is not a certified medical "
            "diagnostic device. Model outputs are probabilistic statistical estimates and must never replace "
            "professional medical evaluation, diagnosis, or treatment decisions."
        )

    # Sidebar: thresholds and active model
    chosen_model_name = None
    with st.sidebar:
        st.markdown("**Risk thresholds**")
        t_low = st.slider("Low to moderate", 0.10, 0.50, float(config.LOW_RISK_THRESHOLD), 0.05)
        t_high = st.slider("Moderate to elevated", 0.50, 0.85, float(config.HIGH_RISK_THRESHOLD), 0.05)
        if model_ok:
            available = list_available_models(pipeline_dict)
            names = [m["name"] for m in available]
            default_idx = next((i for i, m in enumerate(available) if m["is_default"]), 0)
            st.markdown("**Model**")
            chosen_model_name = st.selectbox("Score patients using", names, index=default_idx,
                                             label_visibility="collapsed")
            active = next(m for m in available if m["name"] == chosen_model_name)
            st.markdown("**Selected model performance**")
            st.metric("ROC-AUC", f"{active['metrics'].get('ROC_AUC', 0.0):.3f}")
            st.metric("Sensitivity", f"{active['metrics'].get('Sensitivity_Recall', 0.0):.3f}")
            st.metric("Specificity", f"{active['metrics'].get('Specificity', 0.0):.3f}")

    tab_calc, tab_compare, tab_batch, tab_models, tab_shap, tab_oai = st.tabs([
        "Assessment", "Comparison", "Batch screening",
        "Model performance", "Explainability", "Research data",
    ])

    # =======================================================================
    # Assessment
    # =======================================================================
    with tab_calc:
        if "age_val" not in st.session_state:
            st.session_state.update(age_val=63, sex_val=2, bmi_val=28.5, pain_val=6, dis_val=18,
                                    stiff_val=2, pase_val=145.0, kl_val=2, inj_val=0, surg_val=0)

        section("Patient baseline", "Enter baseline measures, or start from a preset profile.")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("Low-risk profile", use_container_width=True):
                st.session_state.update(age_val=52, sex_val=1, bmi_val=23.2, pain_val=1, dis_val=4,
                                        stiff_val=1, pase_val=210.0, kl_val=0, inj_val=0, surg_val=0)
                st.rerun()
        with preset_cols[1]:
            if st.button("Moderate-risk profile", use_container_width=True):
                st.session_state.update(age_val=63, sex_val=2, bmi_val=28.4, pain_val=6, dis_val=16,
                                        stiff_val=2, pase_val=140.0, kl_val=1, inj_val=1, surg_val=0)
                st.rerun()
        with preset_cols[2]:
            if st.button("High-risk profile", use_container_width=True):
                st.session_state.update(age_val=72, sex_val=2, bmi_val=34.6, pain_val=14, dis_val=38,
                                        stiff_val=5, pase_val=80.0, kl_val=3, inj_val=1, surg_val=1)
                st.rerun()
        with preset_cols[3]:
            if st.button("Random profile", use_container_width=True):
                np.random.seed(int(datetime.now().timestamp()) % 10000)
                st.session_state.update(
                    age_val=int(np.random.randint(48, 79)), sex_val=int(np.random.choice([1, 2])),
                    bmi_val=round(float(np.random.uniform(20.0, 38.0)), 1),
                    pain_val=int(np.random.randint(0, 18)), dis_val=int(np.random.randint(0, 50)),
                    stiff_val=int(np.random.randint(0, 7)),
                    pase_val=round(float(np.random.uniform(50.0, 260.0)), 1),
                    kl_val=int(np.random.choice([0, 1, 2, 3])), inj_val=int(np.random.choice([0, 1])),
                    surg_val=int(np.random.choice([0, 1])),
                )
                st.rerun()

        with st.container(border=True):
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                group("Demographics and biometrics")
                p_age = st.number_input("Age (years)", 40, 90, st.session_state.age_val, step=1)
                p_sex = st.selectbox("Biological sex", [1, 2], index=0 if st.session_state.sex_val == 1 else 1,
                                     format_func=lambda x: "Male" if x == 1 else "Female")
                p_bmi = st.number_input("BMI (kg/m2)", 15.0, 50.0, float(st.session_state.bmi_val), step=0.1)
            with col_c2:
                group("Baseline symptoms (WOMAC)")
                p_pain = st.slider("Pain (0 to 20)", 0, 20, st.session_state.pain_val)
                p_dis = st.slider("Disability (0 to 68)", 0, 68, st.session_state.dis_val)
                p_stiff = st.slider("Stiffness (0 to 8)", 0, 8, st.session_state.stiff_val)
                p_pase = st.number_input("Physical activity (PASE)", 0.0, 400.0,
                                         float(st.session_state.pase_val), step=5.0)
            with col_c3:
                group("Radiographic and trauma history")
                kl_descriptions = {0: "0: Normal", 1: "1: Doubtful", 2: "2: Minimal OA", 3: "3: Moderate OA"}
                p_kl = st.selectbox("Baseline KL grade", [0, 1, 2, 3], index=st.session_state.kl_val,
                                    format_func=lambda x: kl_descriptions[x])
                p_inj = st.selectbox("Prior knee injury", [0, 1], index=st.session_state.inj_val,
                                     format_func=lambda x: "Yes" if x == 1 else "No")
                p_surg = st.selectbox("Prior knee surgery", [0, 1], index=st.session_state.surg_val,
                                      format_func=lambda x: "Yes" if x == 1 else "No")

        patient_input = {
            "V00AGE": p_age, "V00SEX": p_sex, "V00BMI": p_bmi, "V00WOMKP": p_pain,
            "V00WOMAD": p_dis, "V00WOMST": p_stiff, "V00PASE": p_pase,
            "V00KL": p_kl, "V00INJ": p_inj, "V00SURG": p_surg,
        }

        st.write("")
        if st.button("Calculate progression risk", type="primary", use_container_width=True):
            if pipeline_dict is None:
                st.error("Model pipeline not available. Run `python -m src.train` to train the model first.")
            else:
                with st.spinner("Computing risk estimate"):
                    result = predict_patient_risk(patient_input, pipeline_dict, model_name=chosen_model_name)
                st.session_state.assess_n = st.session_state.get("assess_n", 0) + 1
                st.session_state.assessment = {
                    "n": st.session_state.assess_n,
                    "ts": datetime.now(),
                    "input": dict(patient_input),
                    "prob": result["predicted_probability"],
                    "model_used": result.get("model_used", "Ensemble Model"),
                    "contributions": result.get("feature_contributions", {}),
                    "scale": result.get("contribution_scale", "log_odds"),
                    "model_name": chosen_model_name,
                }

        assess = st.session_state.get("assessment")
        if assess and pipeline_dict is not None:
            render_assessment(assess, pipeline_dict, t_low, t_high, patient_input)

    # =======================================================================
    # Comparison
    # =======================================================================
    with tab_compare:
        section("Patient comparison", "Evaluate how modifying risk factors alters the progression estimate.")
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            with st.container(border=True):
                group("Profile A (baseline)")
                a_age = st.number_input("Age (A)", 40, 90, 65, key="a_age")
                a_bmi = st.number_input("BMI (A)", 15.0, 50.0, 32.4, step=0.1, key="a_bmi")
                a_pain = st.slider("WOMAC pain (A)", 0, 20, 10, key="a_pain")
                a_kl = st.selectbox("Baseline KL (A)", [0, 1, 2, 3], index=2, key="a_kl")
                a_inj = st.selectbox("Injury history (A)", [0, 1], index=1,
                                     format_func=lambda x: "Yes" if x == 1 else "No", key="a_inj")
        with cmp_col2:
            with st.container(border=True):
                group("Profile B (intervention)")
                b_age = st.number_input("Age (B)", 40, 90, 65, key="b_age")
                b_bmi = st.number_input("BMI (B)", 15.0, 50.0, 26.0, step=0.1, key="b_bmi")
                b_pain = st.slider("WOMAC pain (B)", 0, 20, 4, key="b_pain")
                b_kl = st.selectbox("Baseline KL (B)", [0, 1, 2, 3], index=2, key="b_kl")
                b_inj = st.selectbox("Injury history (B)", [0, 1], index=1,
                                     format_func=lambda x: "Yes" if x == 1 else "No", key="b_inj")

        st.write("")
        if st.button("Compare progression risk", type="primary", use_container_width=True):
            if pipeline_dict is None:
                st.error("Model pipeline not available. Run `python -m src.train` to train the model first.")
            else:
                pat_a = {"V00AGE": a_age, "V00SEX": 2, "V00BMI": a_bmi, "V00WOMKP": a_pain,
                         "V00WOMAD": a_pain * 2, "V00WOMST": 2, "V00PASE": 120.0,
                         "V00KL": a_kl, "V00INJ": a_inj, "V00SURG": 0}
                pat_b = {"V00AGE": b_age, "V00SEX": 2, "V00BMI": b_bmi, "V00WOMKP": b_pain,
                         "V00WOMAD": b_pain * 2, "V00WOMST": 1, "V00PASE": 160.0,
                         "V00KL": b_kl, "V00INJ": b_inj, "V00SURG": 0}
                prob_a = predict_patient_risk(pat_a, pipeline_dict)["predicted_probability"]
                prob_b = predict_patient_risk(pat_b, pipeline_dict)["predicted_probability"]
                delta_prob = (prob_b - prob_a) * 100

                section("Comparison result")
                render(readout_html("Profile A", prob_a, t_low, t_high))
                render(readout_html("Profile B", prob_b, t_low, t_high))
                cls = "good" if delta_prob < -0.05 else ("bad" if delta_prob > 0.05 else "flat")
                render(
                    f'<div class="delta"><span class="{cls}">Difference (B minus A): '
                    f"<b>{delta_prob:+.1f}</b> percentage points.</span> "
                    f"Risk moved from {prob_a * 100:.1f}% to {prob_b * 100:.1f}%.</div>"
                )
                st.caption("Comparison fixes sex, disability, stiffness, activity, and surgery history at preset values.")

    # =======================================================================
    # Batch screening
    # =======================================================================
    with tab_batch:
        section("Batch cohort screening", "Upload a CSV of multiple patients for batch progression risk scoring.")

        sample_batch_df = pd.DataFrame([
            {"ID": "P001", "V00AGE": 62, "V00SEX": 2, "V00BMI": 31.4, "V00WOMKP": 9, "V00WOMAD": 24, "V00WOMST": 3, "V00PASE": 110.0, "V00KL": 2, "V00INJ": 1, "V00SURG": 0},
            {"ID": "P002", "V00AGE": 54, "V00SEX": 1, "V00BMI": 24.2, "V00WOMKP": 2, "V00WOMAD": 6, "V00WOMST": 1, "V00PASE": 220.0, "V00KL": 0, "V00INJ": 0, "V00SURG": 0},
            {"ID": "P003", "V00AGE": 71, "V00SEX": 2, "V00BMI": 35.8, "V00WOMKP": 14, "V00WOMAD": 38, "V00WOMST": 5, "V00PASE": 75.0, "V00KL": 3, "V00INJ": 1, "V00SURG": 1},
            {"ID": "P004", "V00AGE": 66, "V00SEX": 1, "V00BMI": 27.5, "V00WOMKP": 5, "V00WOMAD": 14, "V00WOMST": 2, "V00PASE": 150.0, "V00KL": 1, "V00INJ": 0, "V00SURG": 0},
        ])
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            csv_sample = sample_batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download template CSV", csv_sample, "oai_batch_template.csv", "text/csv")
        with col_b2:
            uploaded_file = st.file_uploader("Upload patient cohort CSV", type=["csv"])

        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Loaded cohort with {len(df_batch)} patient records.")

            if pipeline_dict is not None:
                preprocessor = pipeline_dict["preprocessor"]
                model = pipeline_dict["model"]
                with st.spinner("Scoring cohort"):
                    probs = model.predict_proba(preprocessor.transform(df_batch))[:, 1]
                    df_batch["Progression_Probability"] = np.round(probs, 4)
                    df_batch["Risk_Tier"] = pd.cut(
                        df_batch["Progression_Probability"], bins=[-0.01, t_low, t_high, 1.01],
                        labels=["Low Risk", "Moderate Risk", "High Risk"],
                    )

                section("Cohort stratification")
                tier_labels = ["Low Risk", "Moderate Risk", "High Risk"]
                counts = df_batch["Risk_Tier"].value_counts().reindex(tier_labels).fillna(0)
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    with st.container(border=True):
                        group("Patients per tier")
                        fig_bar = go.Figure(go.Bar(
                            x=tier_labels, y=counts.values, marker_color=[LOW_C, MOD_C, HIGH_C],
                            text=[int(c) for c in counts.values], textposition="outside", cliponaxis=False,
                        ))
                        fig_bar.update_yaxes(title="Patients")
                        st.plotly_chart(style_fig(fig_bar, height=300, margin=dict(l=10, r=10, t=20, b=30)),
                                        use_container_width=True)
                with stat_col2:
                    with st.container(border=True):
                        group("Probability distribution")
                        fig_hist = px.histogram(df_batch, x="Progression_Probability", nbins=15,
                                                color_discrete_sequence=[ACCENT])
                        fig_hist.update_xaxes(title="Progression probability")
                        fig_hist.update_yaxes(title="Patients")
                        st.plotly_chart(style_fig(fig_hist, height=300, margin=dict(l=10, r=10, t=20, b=30)),
                                        use_container_width=True)

                section("Annotated cohort predictions")
                st.dataframe(df_batch, use_container_width=True)
                annotated_csv = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button("Export annotated predictions (.csv)", annotated_csv,
                                   "oai_predictions_output.csv", "text/csv")

    # =======================================================================
    # Model performance
    # =======================================================================
    with tab_models:
        section("Model comparison and evaluation metrics")
        eval_report_path = config.RESULTS_DIR / "evaluation_report.md"
        if eval_report_path.exists():
            with st.container(border=True):
                with open(eval_report_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

        section("Diagnostic plots")
        fig_files = list(config.FIGURES_DIR.glob("*.png"))
        if fig_files:
            fig_cols = st.columns(len(fig_files))
            for i, fp in enumerate(fig_files):
                with fig_cols[i]:
                    st.image(str(fp), caption=fp.stem.replace("_", " ").title(), use_container_width=True)
        else:
            st.caption("No diagnostic plots found. Run the training pipeline to generate them.")

        section("Decision threshold simulator", "Illustrative only. Values are computed from a fixed formula, not from the trained model.")
        sim_thresh = st.slider("Simulated decision threshold", 0.10, 0.90, 0.50, 0.05)
        sim_sens = max(0.20, min(0.98, 0.85 - (sim_thresh - 0.3) * 0.9))
        sim_spec = max(0.20, min(0.98, 0.55 + (sim_thresh - 0.3) * 0.8))
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Simulated sensitivity", f"{sim_sens:.2%}", delta=f"{(sim_sens - 0.74) * 100:+.1f}% vs default")
        with sc2:
            st.metric("Simulated specificity", f"{sim_spec:.2%}", delta=f"{(sim_spec - 0.76) * 100:+.1f}% vs default")
        with sc3:
            st.caption("Lowering the threshold prioritizes catching every potential progressor, at the cost of more false alarms.")

    # =======================================================================
    # Explainability
    # =======================================================================
    with tab_shap:
        section("Explainability and biomarker context")
        shap_img = config.FIGURES_DIR / "shap_summary_plot.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="Global SHAP summary plot", use_container_width=True)

        with st.container(border=True):
            group("Clinical relevance")
            st.markdown("""
| Feature | Biological name | Mechanism in OA progression |
| --- | --- | --- |
| `V00KL` | Baseline Kellgren-Lawrence grade | Pre-existing chondrocyte depletion, subchondral sclerosis, and marginal osteophytes create an unstable mechanical joint. |
| `V00BMI` | Body mass index | Excess adiposity exerts chronic joint compressive loading and systemic release of pro-inflammatory adipokines. |
| `V00AGE` | Patient age | Aging cartilage exhibits reduced proteoglycan content and impaired repair mechanisms. |
| `V00WOMKP` | WOMAC pain score | Ongoing nociceptive activation often corresponds with synovitis and micro-instability. |
| `V00INJ` | Prior joint injury | Prior ligamentous or meniscal tears cause post-traumatic osteoarthritis via joint incongruence. |
| `V00PASE` | Physical activity | Moderate cyclic loading maintains synovial fluid circulation; extremes can affect cartilage nutrition. |
""")

    # =======================================================================
    # Research data
    # =======================================================================
    with tab_oai:
        section("Osteoarthritis Initiative dataset inspector")
        if st.button("Scan and refresh data/raw/ directory"):
            from src.inspect_dataset import run_stage_1_inspection
            run_stage_1_inspection()
            st.success("Refreshed Stage 1 schema inspection.")

        schema_report = config.RESULTS_DIR / "oai_schema_report.md"
        if schema_report.exists():
            with st.container(border=True):
                with open(schema_report, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
        else:
            st.info("No schema report generated yet. Run `python -m src.inspect_dataset` to profile your files.")


if __name__ == "__main__":
    main()