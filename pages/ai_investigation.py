"""
pages/ai_investigation.py — Technical Diagnostics & Root Cause Analysis
Grounds characterization anomalies against engineering specifications and standards.
"""

import streamlit as st
import polars as pl

from src.rag_engine import RAGEngine
from src.insight_generator import generate_findings

NXP_ORANGE = "#ea7600"
NXP_BLUE   = "#0096d6"
NXP_GREEN  = "#2ea043"
NXP_RED    = "#da3633"

PRESET_INQUIRIES = [
    ("Bimodal Load Modulation Diagnostics", "What could cause a bimodal load modulation distribution?"),
    ("Receiver Sensitivity Systematic Drift", "What should an engineer investigate when a lot shows an unusual shift in receiver sensitivity?"),
    ("Process Margin & Capability (Cp/Cpk)", "What is Cp and Cpk and what do they indicate in semiconductor characterization?"),
    ("Carrier Frequency Measurement & Drift", "How is carrier frequency measured in NFC testing and what causes drift?"),
]


@st.cache_resource(show_spinner="Initializing technical document index…")
def _get_rag_engine():
    return RAGEngine()


def render(df: pl.DataFrame, specs: dict):
    st.markdown("<div class='page-title'>Technical Diagnostics &amp; Root Cause Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Cross-referencing parametric anomalies against engineering specifications and characterization standards</div>",
        unsafe_allow_html=True,
    )

    rag = _get_rag_engine()

    # System Status Bar
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        if rag.llm_available:
            st.markdown(
                "<span class='badge badge-pass'>STATUS: LLM SYNTHESIS ACTIVE (GEMINI 3.5 FLASH)</span> "
                "<span style='color:#8b949e;font-size:0.75rem;margin-left:0.5rem;'>Grounded inference against 8 technical standards</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='badge badge-info'>STATUS: LOCAL DETERMINISTIC RETRIEVAL ACTIVE (BM25)</span> "
                "<span style='color:#8b949e;font-size:0.75rem;margin-left:0.5rem;'>Direct standard clause extraction</span>",
                unsafe_allow_html=True,
            )
    with sc2:
        if not rag.llm_available:
            with st.popover("API Key Configuration", use_container_width=True):
                st.markdown(
                    "<span style='color:#8b949e;font-size:0.8rem;'>"
                    "Enter a Google API key to enable generative synthesis, or configure <code>GOOGLE_API_KEY</code> in Streamlit Secrets."
                    "</span>",
                    unsafe_allow_html=True,
                )
                input_k = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...", label_visibility="collapsed")
                if st.button("Apply Key", use_container_width=True):
                    if input_k.strip():
                        st.session_state["custom_gemini_api_key"] = input_k.strip()
                        rag._init_gemini(input_k.strip())
                        st.rerun()

    st.markdown("<hr style='border-color:#30363d;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────
    main_col, ctx_col = st.columns([3, 1])

    with ctx_col:
        _render_context_panel(df, specs)

    with main_col:
        _render_diagnostic_interface(rag, df, specs)


def _render_context_panel(df: pl.DataFrame, specs: dict):
    st.markdown("<div class='section-header'>Investigation Target</div>", unsafe_allow_html=True)

    all_lots = sorted(df["lot_id"].unique().to_list()) if "lot_id" in df.columns else []
    sel_lot = st.selectbox(
        "Focus Lot",
        ["None"] + all_lots,
        key="ai_ctx_lot",
    )

    if sel_lot != "None":
        lot_df = df.filter(pl.col("lot_id") == sel_lot)
        st.markdown(
            f"""<div class='nxp-card' style='border-left:3px solid {NXP_ORANGE};padding:0.75rem 1rem;'>
                <div style='font-size:0.75rem;font-weight:700;color:#f0f6fc;'>LOT: {sel_lot}</div>
                <div style='color:#8b949e;font-size:0.75rem;font-family:ui-monospace,SFMono-Regular,monospace;margin-top:0.25rem;'>
                    {len(lot_df):,} measurements<br>
                    {lot_df['chip_id'].n_unique()} tested dies<br>
                    {lot_df['wafer_id'].n_unique()} wafers sampled
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        findings = generate_findings(lot_df, specs)
        lot_findings = [f for f in findings if f.lot_id == sel_lot]

        if lot_findings:
            st.markdown("<div style='color:#8b949e;font-size:0.72rem;font-weight:600;text-transform:uppercase;margin-top:0.6rem;margin-bottom:0.3rem;'>Flagged Anomalies:</div>", unsafe_allow_html=True)
            for f in lot_findings[:4]:
                badge = "<span class='badge badge-fail'>CRITICAL</span>" if f.severity == "fail" else "<span class='badge badge-warn'>ANOMALY</span>"
                st.markdown(
                    f"""<div style='font-size:0.76rem;color:#c9d1d9;padding:0.25rem 0;border-bottom:1px solid #21262d;'>
                        {badge} {f.headline.split('—')[-1].strip()}
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<span class='badge badge-pass'>ALL SPECS COMPLIANT</span>", unsafe_allow_html=True)

    # Document Index
    st.markdown("<div class='section-header' style='margin-top:1.2rem;'>Reference Standards</div>", unsafe_allow_html=True)
    rag = _get_rag_engine()
    for doc in rag.documents:
        st.markdown(
            f"""<div style='font-size:0.72rem;color:#8b949e;padding:0.2rem 0;font-family:ui-monospace,SFMono-Regular,monospace;'>
                <span style='color:#58a6ff;'>DOC</span> {doc.title[:30]}
            </div>""",
            unsafe_allow_html=True,
        )


def _render_diagnostic_interface(rag: RAGEngine, df: pl.DataFrame, specs: dict):
    if "ai_question_text" not in st.session_state:
        st.session_state["ai_question_text"] = ""
    if "ai_last_result" not in st.session_state:
        st.session_state["ai_last_result"] = None
    if "ai_last_question" not in st.session_state:
        st.session_state["ai_last_question"] = ""

    # Preset Diagnostic Inquiries
    st.markdown(
        "<div style='font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:0.5rem;'>Standard Diagnostic Inquiries:</div>",
        unsafe_allow_html=True,
    )
    clicked_question = None
    cols = st.columns(2)
    for i, (label, query) in enumerate(PRESET_INQUIRIES):
        with cols[i % 2]:
            if st.button(label, key=f"ai_preset_{i}", use_container_width=True):
                clicked_question = query

    st.markdown("<hr style='border-color:#30363d;margin:0.8rem 0;'>", unsafe_allow_html=True)

    trigger_search = False
    if clicked_question:
        st.session_state["ai_question_text"] = clicked_question
        trigger_search = True

    # Query Input Form
    with st.form("diagnostic_query_form", clear_on_submit=False):
        typed_question = st.text_area(
            "Diagnostic Inquiry / Physical Anomaly Description:",
            value=st.session_state["ai_question_text"],
            height=75,
            placeholder="Describe parametric behavior or inquire about specifications (e.g. Causes of bimodal load modulation amplitude)",
        )
        submit_btn = st.form_submit_button("Run Diagnostic Analysis", use_container_width=True)

    if submit_btn:
        st.session_state["ai_question_text"] = typed_question
        trigger_search = True

    # Build extra context from selected lot
    extra_ctx = ""
    sel_lot = st.session_state.get("ai_ctx_lot", "None")
    if sel_lot and sel_lot != "None":
        lot_df = df.filter(pl.col("lot_id") == sel_lot)
        findings = generate_findings(lot_df, specs)
        lot_findings = [f for f in findings if f.lot_id == sel_lot]
        if lot_findings:
            finding_lines = "\n".join([f"- {f.headline}: {f.detail}" for f in lot_findings[:3]])
            extra_ctx = (
                f"The engineer is investigating lot {sel_lot} with the following characterization findings:\n"
                f"{finding_lines}"
            )

    # Execute Search
    if trigger_search:
        q_to_run = st.session_state["ai_question_text"].strip()
        if q_to_run:
            with st.spinner("Retrieving relevant standards and synthesizing diagnostic report…"):
                st.session_state["ai_last_result"] = rag.answer(q_to_run, extra_context=extra_ctx, k=3)
                st.session_state["ai_last_question"] = q_to_run
        else:
            st.warning("Please enter or select a diagnostic inquiry.")

    # Render Diagnostic Report
    result = st.session_state.get("ai_last_result")
    last_q = st.session_state.get("ai_last_question", "")

    if result and last_q:
        ans_text = result.get("answer")
        mode = result.get("mode")

        st.markdown("""
            <div class='diagnostic-report'>
                <div class='diagnostic-header'>
                    <span>Diagnostic Technical Synthesis</span>
                    <span class='source-chip'>RAG GROUNDED EVALUATION</span>
                </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:0.95rem;font-weight:600;color:#f0f6fc;margin-bottom:0.75rem;'>Inquiry: {last_q}</div>",
            unsafe_allow_html=True,
        )

        if ans_text:
            st.markdown(f"<div class='diagnostic-body'>{ans_text}</div>", unsafe_allow_html=True)

        # Referenced Specifications
        if result.get("retrieved"):
            st.markdown("<div class='section-header' style='margin-top:1.25rem;'>Referenced Technical Clauses &amp; Documents</div>", unsafe_allow_html=True)
            for r in result["retrieved"]:
                with st.expander(f"{r.document.title}  (Relevance: {r.score:.2f})"):
                    st.markdown(f"<div style='font-size:0.83rem;color:#c9d1d9;line-height:1.5;'>{r.excerpt}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-top:0.4rem;'><span class='source-chip'>File: {r.document.source_file}</span></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Synthesis is grounded deterministically in the indexed technical documentation library.")

    elif not result and not trigger_search:
        st.info("Select a preset diagnostic inquiry above or submit a technical description to cross-reference specifications.")


# ── Execution entry point when run by Streamlit ───────────────────
from src.app_context import get_context
_df, _specs = get_context()
render(_df, _specs)
