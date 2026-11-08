"""
pages/statistical_analysis.py  — Page 3: Statistical Analysis
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.statistics_engine import compute_stats, per_lot_stats
from src.multimodal_analysis import fit_gmm, gmm_pdf, component_pdf

NXP_ORANGE = "#ea7600"
NXP_BLUE   = "#0096d6"
NXP_GREEN  = "#2ea043"
NXP_RED    = "#da3633"
PLOT_BG    = "#161b22"
GRID_COLOR = "#21262d"

LOT_COLORS = {
    "LOT-A": NXP_BLUE,
    "LOT-B": NXP_GREEN,
    "LOT-C": NXP_RED,
    "LOT-D": NXP_ORANGE,
    "LOT-E": "#8b5cf6",
}


def _theme():
    return dict(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#c9d1d9", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size=11),
        margin=dict(l=50, r=20, t=35, b=45),
    )


def render(df: pl.DataFrame, specs: dict):
    st.markdown("<div class='page-title'>Statistical Analysis &amp; Process Modelling</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Parametric capability estimation (Cp / Cpk), multi-lot variance, and Gaussian Mixture decomposition</div>",
        unsafe_allow_html=True,
    )

    # ── Selectors ────────────────────────────────────────────────
    sel_col, filt_col = st.columns([2, 3])
    with sel_col:
        all_params = sorted(df["parameter_name"].unique().to_list()) if "parameter_name" in df.columns else []
        sel_param = st.selectbox("Parameter", all_params, key="sa_param")

    with filt_col:
        fc1, fc2 = st.columns(2)
        with fc1:
            all_lots = sorted(df["lot_id"].unique().to_list()) if "lot_id" in df.columns else []
            sel_lots = st.multiselect("Production Lots", all_lots, default=all_lots, key="sa_lots")
        with fc2:
            all_conds = sorted(df["test_condition"].unique().to_list()) if "test_condition" in df.columns else []
            sel_cond  = st.multiselect("Test Condition", all_conds, default=all_conds, key="sa_cond")

    # Filter data
    filtered = df
    if sel_lots and "lot_id" in df.columns:
        filtered = filtered.filter(pl.col("lot_id").is_in(sel_lots))
    if sel_cond and "test_condition" in df.columns:
        filtered = filtered.filter(pl.col("test_condition").is_in(sel_cond))

    param_data = filtered.filter(pl.col("parameter_name") == sel_param) if sel_param else filtered

    if param_data.is_empty():
        st.warning("No data for the selected filters.")
        return

    spec = specs.get(sel_param, {})
    lsl = spec.get("lsl")
    usl = spec.get("usl")
    nominal = spec.get("nominal")
    unit = spec.get("unit", "")

    # ── Stats table ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Descriptive Statistics &amp; Process Indices</div>", unsafe_allow_html=True)
    values = param_data["measured_value"].to_numpy()
    summary = compute_stats(values, lsl=lsl, usl=usl)
    stats_dict = summary.to_dict()

    # Display as a compact horizontal metric row
    stat_items = list(stats_dict.items())
    n_cols = min(len(stat_items), 7)
    cols = st.columns(n_cols)
    for i, (k, v) in enumerate(stat_items[:n_cols]):
        with cols[i]:
            st.metric(k, str(v))
    if len(stat_items) > n_cols:
        cols2 = st.columns(len(stat_items) - n_cols)
        for i, (k, v) in enumerate(stat_items[n_cols:]):
            with cols2[i]:
                st.metric(k, str(v))

    st.markdown("<hr style='border-color:#30363d;margin:1.1rem 0;'>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────
    tabs = st.tabs([
        "Distribution Density",
        "Cross-Lot Variation",
        "Chronological Trend",
        "Parametric Correlation",
        "Multimodal Decomposition",
    ])

    with tabs[0]:
        _render_histogram(param_data, sel_param, unit, lsl, usl, nominal, sel_lots)

    with tabs[1]:
        _render_boxplot(filtered, sel_param, unit, lsl, usl)

    with tabs[2]:
        _render_trend(param_data, sel_param, unit)

    with tabs[3]:
        _render_correlation(filtered, specs)

    with tabs[4]:
        _render_multimodal(param_data, sel_param, unit, sel_lots, lsl, usl)


def _render_histogram(param_data, param, unit, lsl, usl, nominal, sel_lots):
    fig = go.Figure()
    if len(sel_lots) > 1:
        # Overlay per lot
        for lot in sorted(sel_lots):
            sub = param_data.filter(pl.col("lot_id") == lot)
            if sub.is_empty():
                continue
            fig.add_trace(go.Histogram(
                x=sub["measured_value"].to_numpy(),
                name=lot,
                opacity=0.6,
                nbinsx=35,
                marker_color=LOT_COLORS.get(lot, NXP_BLUE),
                hovertemplate=f"{lot}<br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(
            x=param_data["measured_value"].to_numpy(),
            nbinsx=40,
            marker_color=NXP_BLUE,
            marker_line_color="#161b22",
            marker_line_width=0.4,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
            name=param,
        ))

    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"LSL {lsl}", annotation_font_color=NXP_RED,
                      annotation_font_size=10)
    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"USL {usl}", annotation_font_color=NXP_RED,
                      annotation_font_size=10)
    if nominal is not None:
        fig.add_vline(x=nominal, line_dash="dot", line_color=NXP_GREEN,
                      annotation_text=f"Nom {nominal}", annotation_font_color=NXP_GREEN,
                      annotation_font_size=10)

    fig.update_layout(
        **_theme(),
        xaxis_title=f"{param} ({unit})",
        yaxis_title="Count",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_boxplot(filtered, param, unit, lsl, usl):
    fig = go.Figure()
    all_lots = sorted(filtered["lot_id"].unique().to_list()) if "lot_id" in filtered.columns else []

    for lot in all_lots:
        sub = filtered.filter(
            (pl.col("lot_id") == lot) & (pl.col("parameter_name") == param)
        )
        if sub.is_empty():
            continue
        fig.add_trace(go.Box(
            y=sub["measured_value"].to_numpy(),
            name=lot,
            marker_color=LOT_COLORS.get(lot, NXP_BLUE),
            line_color=LOT_COLORS.get(lot, NXP_BLUE),
            boxmean=True,
            hovertemplate=f"{lot}<br>Value: %{{y:.4f}} {unit}<extra></extra>",
        ))

    if lsl is not None:
        fig.add_hline(y=lsl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"LSL {lsl}", annotation_font_color=NXP_RED,
                      annotation_font_size=10)
    if usl is not None:
        fig.add_hline(y=usl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"USL {usl}", annotation_font_color=NXP_RED,
                      annotation_font_size=10)

    fig.update_layout(
        **_theme(),
        yaxis_title=f"{param} ({unit})",
        xaxis_title="Production Lot",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trend(param_data, param, unit):
    if "test_timestamp" not in param_data.columns:
        st.info("No timestamp column available for trend analysis.")
        return

    # Sort by timestamp
    sorted_data = param_data.sort("test_timestamp")

    fig = go.Figure()
    all_lots = sorted(param_data["lot_id"].unique().to_list()) if "lot_id" in param_data.columns else ["All"]

    for lot in all_lots:
        sub = sorted_data.filter(pl.col("lot_id") == lot)
        if sub.is_empty():
            continue
        timestamps = sub["test_timestamp"].to_list()
        values = sub["measured_value"].to_numpy()

        fig.add_trace(go.Scatter(
            x=timestamps, y=values,
            mode="markers",
            name=lot,
            marker=dict(color=LOT_COLORS.get(lot, NXP_BLUE), size=4, opacity=0.6),
            hovertemplate=f"{lot}<br>%{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
        ))

        # Rolling mean (window=20)
        if len(values) >= 20:
            rolling = pd.Series(values).rolling(20, center=True).mean().values
            fig.add_trace(go.Scatter(
                x=timestamps, y=rolling,
                mode="lines",
                name=f"{lot} (rolling mean)",
                line=dict(color=LOT_COLORS.get(lot, NXP_BLUE), width=2),
                showlegend=False,
            ))

    fig.update_layout(
        **_theme(),
        xaxis_title="Test Timestamp",
        yaxis_title=f"{param} ({unit})",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation(filtered, specs):
    # Pick numeric columns for correlation
    num_cols = ["measured_value", "temperature_c", "supply_voltage_v"]
    available = [c for c in num_cols if c in filtered.columns]

    if len(available) < 2:
        st.info("Not enough numeric columns for correlation analysis.")
        return

    # Pivot: one row per chip/condition, columns per parameter
    pivot_rows = {}
    for param in specs.keys():
        sub = filtered.filter(pl.col("parameter_name") == param)
        if sub.is_empty():
            continue
        # Use test_id as the join key
        for row in sub.iter_rows(named=True):
            key = row.get("test_id", "")
            if key not in pivot_rows:
                pivot_rows[key] = {
                    "temperature_c": row.get("temperature_c", np.nan),
                    "supply_voltage_v": row.get("supply_voltage_v", np.nan),
                }
            pivot_rows[key][param] = row["measured_value"]

    if not pivot_rows:
        st.info("No data available for correlation.")
        return

    pivot_df = pd.DataFrame.from_dict(pivot_rows, orient="index").dropna()
    if pivot_df.shape[0] < 5:
        st.info("Not enough data points for correlation analysis after pivot.")
        return

    corr = pivot_df.corr(numeric_only=True)

    labels = [
        c.replace("_", " ").replace("mhz", "MHz").replace("mv", "mV")
         .replace("dbm", "dBm").replace("pct", "%").title()
        for c in corr.columns
    ]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[
            [0.0,  "#da3633"],
            [0.5,  "#21262d"],
            [1.0,  "#0096d6"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color="#c9d1d9")),
    ))
    fig.update_layout(
        **_theme(),
        height=380,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Pearson correlation matrix. Values closer to ±1 indicate stronger linear relationships.")


def _render_multimodal(param_data, param, unit, sel_lots, lsl, usl):
    st.markdown(
        """<p style='color:#8b949e;font-size:0.83rem;'>
        Gaussian Mixture Model (GMM) decomposition compares 1- and 2-component Gaussian fits.
        A lower Bayesian Information Criterion (ΔBIC &gt; 10) indicates statistically significant population splitting.
        </p>""",
        unsafe_allow_html=True,
    )

    sel_lot = st.selectbox(
        "Analyse lot", ["All lots combined"] + sorted(sel_lots),
        key="sa_mm_lot"
    )

    if sel_lot == "All lots combined":
        analysis_data = param_data
        lot_label = "All selected lots"
    else:
        analysis_data = param_data.filter(pl.col("lot_id") == sel_lot)
        lot_label = sel_lot

    if analysis_data.is_empty():
        st.warning("No data for the selected lot.")
        return

    values = analysis_data["measured_value"].drop_nulls().to_numpy()

    with st.spinner("Fitting GMM…"):
        result = fit_gmm(values)

    # BIC comparison
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("BIC (1 component)", f"{result.bic_1:.1f}")
    with b2:
        st.metric("BIC (2 components)", f"{result.bic_2:.1f}")
    with b3:
        delta = result.bic_1 - result.bic_2
        st.metric("ΔBIC (2 wins if >10)", f"{delta:+.1f}",
                  delta="Bimodal" if result.is_bimodal else "Unimodal")

    # ── Chart ────────────────────────────────────────────────────
    fig = go.Figure()

    # Histogram (normalized to density)
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=40,
        histnorm="probability density",
        marker_color=NXP_BLUE,
        marker_line_color="#161b22",
        marker_line_width=0.4,
        opacity=0.5,
        name="Measurements",
    ))

    # GMM overlay
    x_lin = np.linspace(values.min() - 0.5*np.std(values),
                         values.max() + 0.5*np.std(values), 300)

    if result.is_bimodal:
        # Component 1
        y_c0 = component_pdf(x_lin, result, 0)
        y_c1 = component_pdf(x_lin, result, 1)
        fig.add_trace(go.Scatter(
            x=x_lin, y=y_c0,
            mode="lines",
            line=dict(color=NXP_ORANGE, width=2, dash="dot"),
            name=f"Population 1 (μ={result.means[0]:.3f}, {result.weights[0]*100:.0f}%)",
        ))
        fig.add_trace(go.Scatter(
            x=x_lin, y=y_c1,
            mode="lines",
            line=dict(color=NXP_GREEN, width=2, dash="dot"),
            name=f"Population 2 (μ={result.means[1]:.3f}, {result.weights[1]*100:.0f}%)",
        ))

    # Full mixture PDF
    y_mix = gmm_pdf(x_lin, result)
    fig.add_trace(go.Scatter(
        x=x_lin, y=y_mix,
        mode="lines",
        line=dict(color="#ffffff", width=2),
        name="GMM mixture",
    ))

    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"LSL {lsl}", annotation_font_color=NXP_RED,
                      annotation_font_size=9)
    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color=NXP_RED,
                      annotation_text=f"USL {usl}", annotation_font_color=NXP_RED,
                      annotation_font_size=9)

    fig.update_layout(
        **_theme(),
        xaxis_title=f"{param} ({unit})",
        yaxis_title="Probability Density",
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(size=10)),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Auto insight ──────────────────────────────────────────────
    param_labels = {
        "carrier_frequency_mhz": "Carrier Frequency",
        "load_modulation_mv": "Load Modulation Amplitude",
        "receiver_sensitivity_dbm": "Receiver Sensitivity",
        "modulation_depth_pct": "Modulation Depth",
    }
    insight = result.insight_text(param_labels.get(param, param), lot_label)

    border_color = NXP_ORANGE if result.is_bimodal else NXP_BLUE
    badge = "<span class='badge badge-warn'>BIMODAL SPLIT</span>" if result.is_bimodal else "<span class='badge badge-info'>UNIMODAL BASELINE</span>"
    st.markdown(
        f"""<div class='nxp-card' style='border-left:3px solid {border_color};margin-top:0.8rem;'>
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">
                {badge}
                <strong style='color:#f0f6fc;font-size:0.85rem;'>GMM Decomposition Assessment</strong>
            </div>
            <span style='color:#c9d1d9;font-size:0.84rem;'>{insight}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Execution entry point when run by Streamlit ───────────────────
from src.app_context import get_context
_df, _specs = get_context()
render(_df, _specs)
