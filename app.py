"""
app.py — NFC Characterization Workbench
Unified Overview & Methodology Dashboard

DISCLAIMER: Independent educational prototype. Not affiliated with NXP Semiconductors.
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from src.app_context import get_context
from src.insight_generator import generate_findings, compute_overall_pass_rate, SEVERITY_ICON
from src.specification_validator import compute_lot_parameter_matrix, validate_dataframe

NXP_ORANGE  = "#FCA600"
NXP_BLUE    = "#0EAFE0"
NXP_GREEN   = "#69CA00"
NXP_RED     = "#ef4444"
PLOT_BG     = "#0d0d1e"
GRID_COLOR  = "#1e1e35"


def _plotly_theme() -> dict:
    return dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#c8c8e0", family="Inter, Segoe UI, sans-serif", size=12),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=40, r=20, t=40, b=40),
    )


def main():
    df, specs = get_context()

    # ── Header ───────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom:0.4rem;">
            <span style="font-size:2.2rem;font-weight:800;letter-spacing:0.02em;">
                <span style="color:#FCA600;">N</span><span style="color:#0EAFE0;">X</span><span style="color:#69CA00;">P</span>
                <span style="color:#f0f0ff;"> NFC Characterization Workbench</span>
            </span>
        </div>
        <p style="color:#8888aa;font-size:0.92rem;margin-top:0;margin-bottom:1.4rem;">
            Automated Semiconductor Characterization &amp; Statistical Analysis
        </p>
    """, unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────
    total_measurements = len(df)
    n_chips = df["chip_id"].n_unique() if "chip_id" in df.columns else 0
    n_lots  = df["lot_id"].n_unique() if "lot_id" in df.columns else 0
    pass_rate = compute_overall_pass_rate(df, specs)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Measurements", f"{total_measurements:,}")
    with c2:
        st.metric("Chips Tested", f"{n_chips:,}")
    with c3:
        st.metric("Production Lots", f"{n_lots}")
    with c4:
        delta_color = "normal" if pass_rate >= 0.99 else "inverse"
        st.metric(
            "Overall Pass Rate",
            f"{pass_rate*100:.2f}%",
            delta=f"{'✓ All within spec' if pass_rate == 1.0 else f'{(1-pass_rate)*100:.2f}% failing'}",
        )

    st.markdown("<hr style='border-color:#1e1e35;margin:1.3rem 0 1rem 0;'>", unsafe_allow_html=True)

    # ── Unified Tabs: Dashboard vs Methodology ───────────────────
    tab_overview, tab_methodology = st.tabs([
        "📊 Overview Dashboard",
        "📖 Engineering Methodology & Architecture",
    ])

    with tab_overview:
        _render_dashboard(df, specs)

    with tab_methodology:
        _render_methodology()


def _render_dashboard(df: pl.DataFrame, specs: dict):
    # Two column layout
    left, right = st.columns([3, 2])

    with left:
        st.markdown("<div class='section-header'>Production Lot Pass Rate</div>", unsafe_allow_html=True)
        _render_lot_pass_chart(df, specs)

    with right:
        st.markdown("<div class='section-header'>Parameter Distribution</div>", unsafe_allow_html=True)
        param_opts = sorted(df["parameter_name"].unique().to_list()) if "parameter_name" in df.columns else []
        sel_param = st.selectbox(
            "Select parameter",
            param_opts,
            key="overview_param_sel",
            label_visibility="collapsed",
        )
        if sel_param:
            _render_param_hist(df, sel_param, specs)

    st.markdown("<hr style='border-color:#1e1e35;margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Findings
    st.markdown("<div class='section-header'>Automated Engineering Findings</div>", unsafe_allow_html=True)

    with st.spinner("Analysing dataset…"):
        findings = generate_findings(df, specs)

    if not findings:
        st.success("✅ No anomalies detected. All parameters appear within specification and no bimodal signals found.")
    else:
        for f in findings[:8]:
            icon = SEVERITY_ICON.get(f.severity, "•")
            border = {
                "fail":    "finding-card finding-fail",
                "warning": "finding-card finding-warn",
                "info":    "finding-card finding-info",
            }.get(f.severity, "finding-card finding-info")

            st.markdown(
                f"""<div class="{border}">
                    <strong>{icon} {f.headline}</strong><br>
                    <span style="color:#9999bb;font-size:0.84rem;">{f.detail}</span>
                    <br><span style="color:#FCA600;font-size:0.75rem;margin-top:0.3rem;display:inline-block;">
                    → Investigate in: {f.page_link}</span>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_lot_pass_chart(df: pl.DataFrame, specs: dict):
    matrix = compute_lot_parameter_matrix(df, specs)
    lot_summary = (
        matrix
        .group_by("lot_id")
        .agg([
            pl.col("n_total").sum(),
            pl.col("n_pass").sum(),
            pl.col("n_fail").sum(),
        ])
        .with_columns(
            (pl.col("n_pass") / pl.col("n_total") * 100).round(2).alias("pass_rate_pct")
        )
        .sort("lot_id")
    )

    lots       = lot_summary["lot_id"].to_list()
    pass_rates = lot_summary["pass_rate_pct"].to_list()
    colors = [
        NXP_GREEN if r >= 99 else (NXP_ORANGE if r >= 90 else NXP_RED)
        for r in pass_rates
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lots,
        y=pass_rates,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{r:.1f}%" for r in pass_rates],
        textposition="outside",
        textfont=dict(color="#c8c8e0", size=11),
        hovertemplate="<b>%{x}</b><br>Pass Rate: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(
        y=99,
        line_dash="dash",
        line_color=NXP_ORANGE,
        annotation_text="99% threshold",
        annotation_font_color=NXP_ORANGE,
        annotation_font_size=10,
    )
    fig.update_layout(
        **_plotly_theme(),
        yaxis_range=[0, 105],
        yaxis_title="Pass Rate (%)",
        xaxis_title="Production Lot",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def _render_param_hist(df: pl.DataFrame, param: str, specs: dict):
    sub = df.filter(pl.col("parameter_name") == param)
    if sub.is_empty():
        st.info("No data for selected parameter.")
        return

    vals = sub["measured_value"].to_numpy()
    spec = specs.get(param, {})
    lsl = spec.get("lsl")
    usl = spec.get("usl")
    nominal = spec.get("nominal")
    unit = spec.get("unit", "")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals,
        nbinsx=40,
        marker_color=NXP_BLUE,
        marker_line_color="#0a0a14",
        marker_line_width=0.5,
        opacity=0.85,
        hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        name="Measurements",
    ))

    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"LSL {lsl}", annotation_font_color=NXP_RED,
                      annotation_font_size=9)
    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"USL {usl}", annotation_font_color=NXP_RED,
                      annotation_font_size=9)
    if nominal is not None:
        fig.add_vline(x=nominal, line_dash="dot", line_color=NXP_GREEN,
                      annotation_text=f"Nom {nominal}", annotation_font_color=NXP_GREEN,
                      annotation_font_size=9)

    fig.update_layout(
        **_plotly_theme(),
        xaxis_title=f"{param} ({unit})",
        yaxis_title="Count",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig, width="stretch")


def _render_methodology():
    st.markdown("## Motivation & Workflow")
    st.markdown(
        """
This platform demonstrates how software engineering, statistical analysis, and AI can be combined
to support semiconductor product engineers in their daily work of characterizing integrated circuits.

The specific application domain — NFC IC characterization — was chosen because NFC technology
presents several interesting characterization challenges:

- **Multiple operating parameters** that interact with each other and with operating conditions (temperature, voltage corners)
- **Multimodal distributions** that may appear when process or test variability is present
- **Specification compliance** that must be evaluated deterministically, never by statistical inference
- **Rich technical documentation** that engineers need to retrieve quickly during investigations
        """
    )

    st.markdown("## Why Data Preparation Matters")
    st.markdown(
        """
In industrial characterization workflows, raw test data frequently contains:

- **Duplicate records** from test retries or data export errors
- **Missing values** where equipment returned no reading
- **Outliers** that may indicate measurement equipment failures (not device failures)
- **Type inconsistencies** when data from multiple sources is merged

Data preparation is not a cosmetic step. Analysing uncleaned data produces misleading statistics.
A histogram that includes test equipment failures (extreme outliers) will obscure the true device distribution.
A pass rate computed on duplicate records will be miscalculated.
The preparation pipeline in this workbench is **transparent**: every change is counted, reported,
and explained. No data is silently discarded.
        """
    )

    st.markdown("## Why Multimodal Distributions Matter")
    st.markdown(
        """
A **unimodal Gaussian distribution** is what semiconductor engineers expect from a well-controlled process:
a single peak centred near the nominal design target, with a spread determined by natural process variation.

A **bimodal distribution** (two peaks) is a significant deviation from this expectation. It suggests that
the measurement population can be separated into two groups with different behaviour. Possible causes include:

- Two wafer lots merged into a single production lot without proper traceability
- A bimodal manufacturing parameter (e.g., etch rate with two operating regimes)
- Mixed test conditions inadvertently combined in the dataset
- Two distinct test sites with a systematic offset between them
- A design discontinuity (e.g., a configuration register with two states)

The key engineering point is: **a bimodal distribution is an investigation signal, not a confirmed defect**.
If both populations remain within specification, the chips may be functionally compliant.
Understanding the root cause of two distinct populations is essential for process reliability and long-term yield stability.
        """
    )

    st.markdown("## Why Deterministic Specification Validation is Essential")
    st.markdown(
        """
In semiconductor engineering, **specification limits are contractual commitments** to customers.
A chip is either within specification or it is not. This is a **binary, deterministic decision**
based on the measured value and the defined limits.

The NFC Characterization Workbench enforces this principle strictly:
- **Pass/fail decisions are made exclusively by Python code**, comparing measured values to spec limits from the JSON file
- **An LLM is never involved in pass/fail decisions**, specification limit definition, or threshold modification
- Statistical anomalies (e.g., bimodal distributions) are reported separately from specification violations
        """
    )

    st.markdown("## How Semantic Retrieval (RAG) Complements Statistical Analysis")
    st.markdown(
        """
When a statistical anomaly is detected — for example, a bimodal distribution in load modulation —
the engineer's next step is to understand what might cause this pattern.

Retrieval-Augmented Generation (RAG) accelerates this by:
1. **Indexing** available technical documents (NFC standards, test procedures, failure guides)
2. **Retrieving** the most relevant passages for the engineer's specific question
3. **Synthesising** a concise answer grounded in the retrieved context using Google Gemini
4. **Citing sources** so the engineer can independently verify the information
        """
    )

    st.markdown("## System Architecture")
    st.code(
        """
Characterization Test Data (CSV)
    │
    ▼
data_generator.py ──── Synthetic dataset (5 000 records, fixed seed=42)
    │
    ▼
data_processor.py ──── Polars cleaning pipeline
    │                   (dedup, null handling, type validation, outlier flagging)
    │
    ├──► statistics_engine.py ──── Descriptive stats, Cp/Cpk
    │
    ├──► multimodal_analysis.py ── scikit-learn GaussianMixture (BIC comparison)
    │
    └──► specification_validator.py ── Deterministic pass/fail engine
            │
            ▼
        insight_generator.py ── Rule-based findings (shifts, bimodality, failures)
            │
            ▼
        Streamlit UI ──── Plotly interactive charts & Polars tables

Technical Documentation (*.txt) ──► BM25 Retrieval ──► Google Gemini ──► AI Investigation
        """,
        language="text",
    )

    st.markdown("## Technology Stack")
    tech_rows = [
        ("Python 3.13", "Runtime environment"),
        ("Streamlit", "Interactive web application framework"),
        ("Polars", "High-performance vectorized data processing"),
        ("NumPy & SciPy", "Descriptive statistics, capability indices, distributions"),
        ("Plotly", "Interactive charts with dark engineering themes"),
        ("scikit-learn", "GaussianMixture model for bimodal detection"),
        ("Google GenAI SDK", "Gemini LLM for grounded technical explanations"),
        ("BM25 Search", "Zero-download local document retrieval engine"),
        ("JSON", "Editable engineering specification limits"),
    ]
    st.dataframe(
        pd.DataFrame(tech_rows, columns=["Component", "Role"]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("## 90-Second Interview Demonstration Script")
    st.markdown(
        r"""
1. **Overview & Findings (25s)**: Point to the KPI cards and the production lot pass rate chart. Note that `LOT-C` shows a systematic degradation on Receiver Sensitivity, and `LOT-D` triggers an automated finding for a bimodal distribution in Load Modulation.
2. **Data Explorer (20s)**: Filter across temperature/voltage corners, show the data cleaning pipeline audit (duplicate and null tracking), and demonstrate the CSV export.
3. **Statistical Analysis & GMM (25s)**: Select `load_modulation_mv` and `LOT-D`. Switch to the **Multimodal** tab to show the two distinct bell curves fitted by the Gaussian Mixture Model ($\Delta\text{BIC} = 104.3$).
4. **AI Investigation (20s)**: Ask: *"What could cause a bimodal distribution in load modulation?"* Show the Gemini-generated explanation grounded in the retrieved documentation with cited sources.
        """
    )
    st.markdown("<hr style='border-color:#1e1e35;margin:2rem 0 1rem;'>", unsafe_allow_html=True)
    st.caption("All datasets, test records, and reference documents in this workbench are synthetic.")


if __name__ == "__main__":
    main()
