"""
pages/data_explorer.py  — Page 2: Data Explorer
"""

import streamlit as st
import polars as pl
import io

from src.data_processor import process

NXP_ORANGE = "#FCA600"
NXP_BLUE   = "#0EAFE0"
NXP_GREEN  = "#69CA00"


def render(df: pl.DataFrame, specs: dict):
    st.markdown("<div class='page-title'>Parametric Data Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Vectorized record inspection, corner-condition filtering, and ingestion pipeline audit</div>",
        unsafe_allow_html=True,
    )

    tab_browse, tab_prep = st.tabs(["Parametric Records", "Data Preparation Pipeline Audit"])

    with tab_browse:
        _render_browse(df)

    with tab_prep:
        _render_preparation(df)


def _render_browse(df: pl.DataFrame):
    st.markdown("<div class='section-header'>Dataset Dimensions</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{len(df.columns)}")
    with c3:
        n_params = df["parameter_name"].n_unique() if "parameter_name" in df.columns else 0
        st.metric("Parameters", n_params)
    with c4:
        n_lots = df["lot_id"].n_unique() if "lot_id" in df.columns else 0
        st.metric("Lots", n_lots)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Filters</div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        all_lots = sorted(df["lot_id"].unique().to_list()) if "lot_id" in df.columns else []
        sel_lots = st.multiselect("Production Lots", all_lots, default=all_lots,
                                   key="de_lots")
    with fc2:
        all_params = sorted(df["parameter_name"].unique().to_list()) if "parameter_name" in df.columns else []
        sel_params = st.multiselect("Parameters", all_params, default=all_params,
                                     key="de_params")
    with fc3:
        all_sites = sorted(df["test_site"].unique().to_list()) if "test_site" in df.columns else []
        sel_sites = st.multiselect("Test Sites", all_sites, default=all_sites,
                                    key="de_sites")

    fc4, fc5 = st.columns(2)
    with fc4:
        if "temperature_c" in df.columns:
            temps = sorted(df["temperature_c"].unique().to_list())
            sel_temps = st.multiselect("Temperature (°C)", temps, default=temps, key="de_temps")
        else:
            sel_temps = []
    with fc5:
        if "supply_voltage_v" in df.columns:
            voltages = sorted(df["supply_voltage_v"].unique().to_list())
            sel_volts = st.multiselect("Supply Voltage (V)", voltages, default=voltages, key="de_volts")
        else:
            sel_volts = []

    # Apply filters
    filtered = df
    if sel_lots and "lot_id" in df.columns:
        filtered = filtered.filter(pl.col("lot_id").is_in(sel_lots))
    if sel_params and "parameter_name" in df.columns:
        filtered = filtered.filter(pl.col("parameter_name").is_in(sel_params))
    if sel_sites and "test_site" in df.columns:
        filtered = filtered.filter(pl.col("test_site").is_in(sel_sites))
    if sel_temps and "temperature_c" in df.columns:
        filtered = filtered.filter(pl.col("temperature_c").is_in(sel_temps))
    if sel_volts and "supply_voltage_v" in df.columns:
        filtered = filtered.filter(pl.col("supply_voltage_v").is_in(sel_volts))

    st.markdown(
        f"<span style='color:#6666aa;font-size:0.8rem;'>Showing {len(filtered):,} of {len(df):,} records</span>",
        unsafe_allow_html=True,
    )

    # Pagination
    PAGE_SIZE = 50
    max_page = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1,
                            key="de_page")
    start = (page - 1) * PAGE_SIZE
    page_df = filtered.slice(start, PAGE_SIZE)

    # Convert to pandas for display (Streamlit renders pandas more cleanly)
    st.dataframe(page_df.to_pandas(), use_container_width=True, height=380)

    # Download
    csv_bytes = filtered.write_csv().encode("utf-8")
    st.download_button(
        "Export Filtered Records (CSV)",
        data=csv_bytes,
        file_name="nfc_filtered_data.csv",
        mime="text/csv",
        key="de_download",
    )

    # ── Column info ───────────────────────────────────────────────
    with st.expander("Column information & missing value report"):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_null = int(df[col].is_null().sum())
            n_unique = df[col].n_unique()
            col_info.append({
                "Column": col,
                "Type": dtype,
                "Null Count": n_null,
                "Null %": f"{n_null/len(df)*100:.1f}%" if len(df) > 0 else "0%",
                "Unique Values": n_unique,
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


def _render_preparation(df: pl.DataFrame):
    st.markdown("<div class='section-header'>Data Preparation Pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        """<p style='color:#9999bb;font-size:0.85rem;'>
        The pipeline below runs automatically on every loaded dataset.
        It checks for missing columns, type issues, duplicate records, null measurements,
        infinite values, and extreme statistical outliers.
        No data is silently discarded — all changes are reported below.
        </p>""",
        unsafe_allow_html=True,
    )

    with st.spinner("Running preparation pipeline…"):
        cleaned_df, report = process(df)

    # ── Before / After ────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("""
            <div class='nxp-card nxp-card-blue'>
                <div style='color:#0EAFE0;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>Before</div>
        """, unsafe_allow_html=True)
        st.metric("Records", f"{report.n_input:,}")
        st.markdown("</div>", unsafe_allow_html=True)

    with bc2:
        st.markdown("""
            <div class='nxp-card nxp-card-green'>
                <div style='color:#69CA00;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>After</div>
        """, unsafe_allow_html=True)
        st.metric("Records", f"{report.n_output:,}",
                  delta=f"-{report.n_input - report.n_output} removed")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:1rem;'>Pipeline Steps Report</div>",
                unsafe_allow_html=True)

    steps = [
        ("Column Name Standardisation", f"Renamed: {report.columns_renamed or 'None'}",
         "PASS" if not report.columns_renamed else "INFO"),
        ("Schema Conformance Check",
         f"Missing: {', '.join(report.missing_columns) if report.missing_columns else 'None'}",
         "PASS" if not report.missing_columns else "WARN"),
        ("Numeric Type Casting",
         f"Cast errors: {report.n_type_cast_errors}",
         "PASS" if report.n_type_cast_errors == 0 else "WARN"),
        ("Deduplication Audit",
         f"{report.n_duplicates_removed} duplicate rows purged",
         "PASS" if report.n_duplicates_removed == 0 else "PURGED"),
        ("Null Measurement Filter",
         f"{report.n_null_measured_value} null rows removed",
         "PASS" if report.n_null_measured_value == 0 else "PURGED"),
        ("Infinite Value Removal",
         f"{report.n_inf_flagged} non-finite float rows removed",
         "PASS" if report.n_inf_flagged == 0 else "PURGED"),
        ("Statistical Outlier Purge (>10σ)",
         f"{report.n_out_of_range_flagged} extreme test artifacts purged",
         "PASS" if report.n_out_of_range_flagged == 0 else "PURGED"),
    ]

    import pandas as pd
    step_df = pd.DataFrame(steps, columns=["Pipeline Gate", "Audit Detail", "Gate Status"])
    st.dataframe(step_df, use_container_width=True, hide_index=True)

    if report.warnings:
        with st.expander("Pipeline Warnings & Alerts"):
            for w in report.warnings:
                st.warning(w)
    else:
        st.info("Pipeline audit completed: Clean dataset validated for downstream parametric analysis.")

    # Duplicate check detail
    with st.expander("Duplicate record analysis"):
        n_dupes = report.n_duplicates_removed
        if n_dupes > 0:
            st.warning(f"{n_dupes} exact duplicate rows were detected and removed.")
        else:
            st.success("No duplicate records found.")

    with st.expander("Dataset schema (cleaned)"):
        schema_rows = [
            {"Column": col, "Dtype": str(cleaned_df[col].dtype)}
            for col in cleaned_df.columns
        ]
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)


# ── Execution entry point when run by Streamlit ───────────────────
from src.app_context import get_context
_df, _specs = get_context()
render(_df, _specs)
