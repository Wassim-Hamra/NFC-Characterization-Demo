"""
src/app_context.py
------------------
Shared application state, CSS styling, and sidebar controls.
Ensures every page has data, specifications, NXP branding, and sidebar controls
regardless of how it was navigated to.
"""

import json
import base64
from pathlib import Path
import streamlit as st
import polars as pl
from src.data_generator import load_or_generate
from src.data_processor import validate_upload_schema

def _load_css():
    css_path = Path("assets/styles.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def _svg_to_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()

@st.cache_data
def _load_specs():
    with open("data/specifications.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner="Loading synthetic dataset…")
def _load_default_data():
    return load_or_generate("data/synthetic_measurements.csv")

def get_context():
    """
    Initializes page configuration, loads CSS styles, ensures data/specs exist
    in session state, and renders the persistent NXP sidebar.
    Returns (df, specs).
    """
    try:
        st.set_page_config(
            page_title="NFC Characterization Workbench",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    _load_css()

    # Specifications
    if "specs" not in st.session_state or st.session_state["specs"] is None:
        st.session_state["specs"] = _load_specs()

    # Data
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "synthetic"

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.session_state["df"] = _load_default_data()

    df = st.session_state["df"]
    specs = st.session_state["specs"]

    # Render persistent NXP sidebar branding and data controls
    render_sidebar(df)

    return df, specs

def render_sidebar(df: pl.DataFrame):
    """Render consistent NXP sidebar branding, dataset status, and upload widgets."""
    with st.sidebar:
        # NXP Logo
        logo_b64 = _svg_to_b64("assets/nxp_logo.svg")
        if logo_b64:
            st.markdown(
                f'<img src="data:image/svg+xml;base64,{logo_b64}" style="width:130px;margin-bottom:0.2rem;">',
                unsafe_allow_html=True,
            )
        st.markdown(
            """<div style="font-size:0.68rem;color:#6666aa;margin-bottom:1rem;line-height:1.3;">
            <strong>NFC Characterization Workbench</strong><br>
            Semiconductor Intelligence Platform<br>
            <span style="color:#444477;">Synthetic Educational Prototype</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color:#1e1e35;margin:0.5rem 0 0.8rem 0;'>", unsafe_allow_html=True)

        # Dataset Status
        st.markdown(
            "<span class='badge badge-synth'>● 5,000 synthetic test records active</span>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:0.75rem;color:#8888aa;margin-top:0.4rem;line-height:1.4;'>"
            f"5 lots (LOT-A to LOT-E)<br>4 NFC parameters</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color:#1e1e35;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.62rem;color:#444466;line-height:1.4;'>"
            "Independent prototype.<br>Not affiliated with NXP Semiconductors.<br>"
            "Synthetic data only.</div>",
            unsafe_allow_html=True,
        )
