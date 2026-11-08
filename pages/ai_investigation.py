"""
pages/ai_investigation.py  — Page 5: AI Investigation (RAG-based)
"""

import streamlit as st
import polars as pl

from src.rag_engine import RAGEngine
from src.insight_generator import generate_findings, _param_label

NXP_ORANGE = "#FCA600"
NXP_BLUE   = "#0EAFE0"
NXP_GREEN  = "#69CA00"
NXP_RED    = "#ef4444"

EXAMPLE_QUESTIONS = [
    "What is load modulation and why does it matter?",
    "What could cause a bimodal load modulation distribution?",
    "Which specification applies to receiver sensitivity?",
    "Why might a measurement distribution become bimodal?",
    "What should an engineer investigate when a lot shows an unusual shift in receiver sensitivity?",
    "What are typical causes of a systematic shift in a production lot?",
    "How is carrier frequency measured in NFC testing?",
    "What is Cp and Cpk and what do they indicate?",
]


@st.cache_resource(show_spinner="Loading document index…")
def _get_rag_engine():
    return RAGEngine()


def render(df: pl.DataFrame, specs: dict):
    st.markdown("<div class='page-title'>AI Investigation Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>RAG-assisted root-cause investigation grounded in technical documentation</div>",
        unsafe_allow_html=True,
    )

    rag = _get_rag_engine()

    # LLM status badge
    if rag.llm_available:
        st.markdown(
            f"<span class='badge badge-pass'>● Gemini API connected — LLM answers enabled</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span class='badge badge-synth'>● Retrieval-only mode — Set GOOGLE_API_KEY in .env to enable LLM answers</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1e1e35;margin:1rem 0;'>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────
    main_col, ctx_col = st.columns([3, 1])

    with ctx_col:
        _render_context_panel(df, specs)

    with main_col:
        _render_chat(rag, df, specs)


def _render_context_panel(df: pl.DataFrame, specs: dict):
    st.markdown(
        "<div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#6666aa;border-bottom:1px solid #1e1e35;padding-bottom:0.4rem;margin-bottom:0.8rem;'>"
        "Investigation Context</div>",
        unsafe_allow_html=True,
    )

    sel_lot = st.selectbox(
        "Focus lot",
        ["None"] + sorted(df["lot_id"].unique().to_list()) if "lot_id" in df.columns else ["None"],
        key="ai_ctx_lot",
    )

    if sel_lot != "None":
        lot_df = df.filter(pl.col("lot_id") == sel_lot)
        st.markdown(
            f"<div class='nxp-card nxp-card-orange' style='font-size:0.82rem;'>"
            f"<strong style='color:#FCA600;'>{sel_lot}</strong><br>"
            f"<span style='color:#9999bb;'>{len(lot_df):,} measurements<br>"
            f"{lot_df['chip_id'].n_unique()} chips<br>"
            f"{lot_df['wafer_id'].n_unique()} wafers</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Show findings for this lot
        findings = generate_findings(lot_df, specs)
        lot_findings = [f for f in findings if f.lot_id == sel_lot]

        if lot_findings:
            st.markdown(
                "<div style='color:#FCA600;font-size:0.75rem;margin-top:0.5rem;'>"
                "⚠️ Detected anomalies:</div>",
                unsafe_allow_html=True,
            )
            for f in lot_findings[:4]:
                icon = "🔴" if f.severity == "fail" else "⚠️"
                st.markdown(
                    f"<div style='font-size:0.75rem;color:#9999bb;padding:0.2rem 0;'>"
                    f"{icon} {f.headline.split('—')[-1].strip()}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='color:#69CA00;font-size:0.75rem;margin-top:0.5rem;'>"
                "✅ No anomalies detected</div>",
                unsafe_allow_html=True,
            )

    # Document index info
    st.markdown(
        f"<div style='margin-top:1rem;font-size:0.72rem;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:#6666aa;border-top:1px solid #1e1e35;"
        f"padding-top:0.6rem;'>Document Index</div>",
        unsafe_allow_html=True,
    )
    rag = _get_rag_engine()
    for doc in rag.documents:
        st.markdown(
            f"<div style='font-size:0.75rem;color:#6666aa;padding:0.1rem 0;'>"
            f"📄 {doc.title[:35]}</div>",
            unsafe_allow_html=True,
        )


def _render_chat(rag: RAGEngine, df: pl.DataFrame, specs: dict):
    # Example questions
    st.markdown(
        "<div style='font-size:0.75rem;color:#6666aa;margin-bottom:0.5rem;'>Example questions:</div>",
        unsafe_allow_html=True,
    )
    example_cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS[:4]):
        with example_cols[i % 2]:
            if st.button(f"❓ {q[:55]}…" if len(q) > 55 else f"❓ {q}",
                         key=f"ai_eg_{i}", use_container_width=True):
                st.session_state["ai_question"] = q

    st.markdown("<hr style='border-color:#1e1e35;margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Question input
    question = st.text_area(
        "Ask a question about NFC characterization:",
        value=st.session_state.get("ai_question", ""),
        height=90,
        placeholder="e.g. What could cause a bimodal load modulation distribution?",
        key="ai_question_input",
    )

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
                f"The engineer is investigating lot {sel_lot} which has the following detected anomalies:\n"
                f"{finding_lines}"
            )

    ask_btn = st.button("🔍 Investigate", use_container_width=True, key="ai_ask_btn")

    if ask_btn and question.strip():
        with st.spinner("Retrieving relevant documentation…"):
            result = rag.answer(question.strip(), extra_context=extra_ctx, k=3)

        # Question bubble
        st.markdown(
            f"<div class='chat-bubble-user'><strong>Question:</strong> {question.strip()}</div>",
            unsafe_allow_html=True,
        )

        # Retrieved sources
        if result["retrieved"]:
            st.markdown(
                "<div style='font-size:0.75rem;color:#6666aa;margin:0.6rem 0 0.3rem;'>"
                "📚 Retrieved passages:</div>",
                unsafe_allow_html=True,
            )
            for r in result["retrieved"]:
                with st.expander(
                    f"📄 {r.document.title}  (relevance: {r.score:.2f})",
                    expanded=False,
                ):
                    st.markdown(
                        f"<div style='font-size:0.82rem;color:#9999bb;'>{r.excerpt}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<span class='source-chip'>Source: {r.document.source_file}</span>",
                        unsafe_allow_html=True,
                    )

        # Answer
        if result["mode"] == "llm" and result["answer"]:
            st.markdown(
                "<div style='font-size:0.75rem;color:#69CA00;margin:0.8rem 0 0.3rem;'>"
                "🤖 Gemini answer (grounded in retrieved documentation):</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='chat-bubble-ai'>{result['answer']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "⚠️ This answer was generated by an LLM using only the retrieved synthetic documentation. "
                "It does not reflect real NXP specifications. All specification limits in the documentation "
                "are illustrative only."
            )
        elif result["mode"] == "retrieval_only":
            if result.get("error"):
                st.warning(f"Retrieval-only mode: {result['error']}")
            else:
                st.markdown(
                    "<div style='font-size:0.75rem;color:#0EAFE0;margin:0.8rem 0 0.3rem;'>"
                    "📖 Retrieval-only mode — most relevant passages shown above</div>",
                    unsafe_allow_html=True,
                )
                st.info(
                    "To enable LLM-synthesised answers, add your Gemini API key to the .env file "
                    "as GOOGLE_API_KEY=<your_key>. The retrieval-only mode shows the most relevant "
                    "document passages directly without synthesis."
                )

        # No documents found
        if not result["retrieved"]:
            st.warning(
                "No relevant passages found in the documentation for this question. "
                "Try rephrasing, or the topic may not be covered by the available documents."
            )

    elif ask_btn and not question.strip():
        st.warning("Please enter a question.")

    # Technical note
    st.markdown("<hr style='border-color:#1e1e35;margin:1.5rem 0 0.5rem;'>", unsafe_allow_html=True)
    st.caption(
        "📌 Answers are synthesized strictly from retrieved technical reference documentation. "
        "Sources are cited with each response."
    )


# ── Execution entry point when run by Streamlit ───────────────────
from src.app_context import get_context
_df, _specs = get_context()
render(_df, _specs)
