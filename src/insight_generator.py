"""
insight_generator.py
--------------------
Rule-based generation of engineering findings for the Overview dashboard.
No LLM — all logic is deterministic.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass
from src.specification_validator import compute_lot_parameter_matrix
from src.multimodal_analysis import fit_gmm


@dataclass
class Finding:
    severity: str    # "info" | "warning" | "fail"
    lot_id: str
    parameter: str
    headline: str
    detail: str
    page_link: str   # suggested page to investigate


SEVERITY_ICON = {
    "info":    "ℹ️",
    "warning": "⚠️",
    "fail":    "🔴",
}


def generate_findings(df: pl.DataFrame, specs: dict) -> list[Finding]:
    """
    Scan the dataset and return a list of automatically detected findings.
    """
    findings: list[Finding] = []

    # ── Specification violations ──────────────────────────────────
    matrix = compute_lot_parameter_matrix(df, specs)

    for row in matrix.iter_rows(named=True):
        if row["status"] == "FAIL":
            findings.append(Finding(
                severity="fail",
                lot_id=row["lot_id"],
                parameter=row["parameter_name"],
                headline=f"{row['lot_id']} — Specification FAIL: {_param_label(row['parameter_name'])}",
                detail=(
                    f"{row['n_fail']} out of {row['n_total']} measurements fail the specification "
                    f"for **{_param_label(row['parameter_name'])}** in lot **{row['lot_id']}** "
                    f"(pass rate: {row['pass_rate']:.1f}%). "
                    f"This lot may require engineering review and disposition."
                ),
                page_link="Statistical Analysis",
            ))
        elif row["status"] == "WARNING":
            findings.append(Finding(
                severity="warning",
                lot_id=row["lot_id"],
                parameter=row["parameter_name"],
                headline=f"{row['lot_id']} — Warning: {_param_label(row['parameter_name'])} near limit",
                detail=(
                    f"{row['n_fail']} measurement(s) outside specification for "
                    f"**{_param_label(row['parameter_name'])}** in lot **{row['lot_id']}** "
                    f"(pass rate: {row['pass_rate']:.1f}%). Monitor this parameter closely."
                ),
                page_link="Statistical Analysis",
            ))

    # ── Bimodal distribution detection ───────────────────────────
    for param_name in specs.keys():
        for lot in sorted(df["lot_id"].unique().to_list()):
            sub = df.filter(
                (pl.col("lot_id") == lot) & (pl.col("parameter_name") == param_name)
            )
            if sub.height < 30:
                continue
            values = sub["measured_value"].to_numpy()
            try:
                result = fit_gmm(values)
                if result.is_bimodal:
                    sep = abs(result.means[1] - result.means[0])
                    findings.append(Finding(
                        severity="warning",
                        lot_id=lot,
                        parameter=param_name,
                        headline=f"{lot} — Bimodal distribution detected: {_param_label(param_name)}",
                        detail=(
                            f"Statistical analysis of **{_param_label(param_name)}** in lot **{lot}** "
                            f"reveals two distinct measurement populations separated by {sep:.2f} units "
                            f"(ΔBIC = {result.bic_improvement:.1f}). "
                            f"This pattern warrants investigation by wafer ID and test site. "
                            f"See the Statistical Analysis page for the GMM decomposition."
                        ),
                        page_link="Statistical Analysis",
                    ))
            except Exception:
                pass

    # ── Systematic shift detection ────────────────────────────────
    for param_name, spec in specs.items():
        nominal = spec["nominal"]
        sigma_threshold = 2.0   # flag if lot mean > 2σ from nominal

        # Compute overall sigma from all lots
        all_vals = df.filter(pl.col("parameter_name") == param_name)["measured_value"].to_numpy()
        if len(all_vals) < 10:
            continue
        overall_std = float(np.std(all_vals, ddof=1))
        if overall_std == 0:
            continue

        for lot in sorted(df["lot_id"].unique().to_list()):
            sub = df.filter(
                (pl.col("lot_id") == lot) & (pl.col("parameter_name") == param_name)
            )
            if sub.height < 10:
                continue
            lot_mean = float(sub["measured_value"].mean())
            z_score = abs(lot_mean - nominal) / overall_std
            if z_score > sigma_threshold:
                direction = "above" if lot_mean > nominal else "below"
                # Only add if not already a FAIL finding for this lot+param
                already_reported = any(
                    f.lot_id == lot and f.parameter == param_name and f.severity == "fail"
                    for f in findings
                )
                if not already_reported:
                    findings.append(Finding(
                        severity="warning",
                        lot_id=lot,
                        parameter=param_name,
                        headline=f"{lot} — Shifted mean: {_param_label(param_name)} ({direction} nominal)",
                        detail=(
                            f"Lot **{lot}** shows a mean of **{lot_mean:.4f}** for "
                            f"**{_param_label(param_name)}** vs. nominal **{nominal}** "
                            f"(z-score = {z_score:.1f}σ). "
                            f"This systematic shift may indicate a process-level deviation."
                        ),
                        page_link="Statistical Analysis",
                    ))

    # Sort: fail first, then warning, then info
    order = {"fail": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.lot_id))
    return findings


def _param_label(param_name: str) -> str:
    labels = {
        "carrier_frequency_mhz": "Carrier Frequency",
        "load_modulation_mv": "Load Modulation Amplitude",
        "receiver_sensitivity_dbm": "Receiver Sensitivity",
        "modulation_depth_pct": "Modulation Depth",
    }
    return labels.get(param_name, param_name)


def compute_overall_pass_rate(df: pl.DataFrame, specs: dict) -> float:
    """Return the fraction of all measurements that pass their spec."""
    from src.specification_validator import validate_dataframe
    enriched = validate_dataframe(df, specs)
    n_total = len(enriched)
    if n_total == 0:
        return 1.0
    n_pass = int(enriched.filter(pl.col("pass_fail") == "PASS").height)
    return n_pass / n_total
