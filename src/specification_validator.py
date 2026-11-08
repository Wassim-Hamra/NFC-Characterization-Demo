"""
specification_validator.py
--------------------------
Deterministic pass/fail validation of measurements against JSON specifications.

IMPORTANT: Pass/fail decisions are made exclusively by this Python code.
An LLM must never modify, invent, or override specification limits.
"""

import json
import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass


SPECS_PATH = Path("data/specifications.json")


def load_specs(path: str | Path = SPECS_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class ParameterValidationResult:
    parameter: str
    unit: str
    lsl: float
    usl: float
    nominal: float
    n_total: int
    n_pass: int
    n_fail: int
    pass_rate: float
    fail_rate: float
    min_measured: float
    max_measured: float
    mean_measured: float
    std_measured: float
    status: str   # "PASS" / "WARNING" / "FAIL"

    @property
    def status_emoji(self) -> str:
        return {"PASS": "🟢", "WARNING": "🟡", "FAIL": "🔴"}.get(self.status, "⚪")


def validate_dataframe(df: pl.DataFrame, specs: dict) -> pl.DataFrame:
    """
    Add pass/fail columns to the DataFrame for each row based on spec limits.
    Returns enriched DataFrame with added columns: lsl, usl, nominal, pass_fail.
    """
    # Build a spec lookup Series for join
    spec_rows = []
    for param_name, s in specs.items():
        spec_rows.append({
            "parameter_name": param_name,
            "spec_lsl": float(s["lsl"]),
            "spec_usl": float(s["usl"]),
            "spec_nominal": float(s["nominal"]),
            "spec_unit": s["unit"],
        })
    spec_df = pl.DataFrame(spec_rows)

    enriched = df.join(spec_df, on="parameter_name", how="left")
    enriched = enriched.with_columns([
        pl.when(
            (pl.col("measured_value") >= pl.col("spec_lsl")) &
            (pl.col("measured_value") <= pl.col("spec_usl"))
        ).then(pl.lit("PASS")).otherwise(pl.lit("FAIL")).alias("pass_fail"),
        (pl.col("measured_value") - pl.col("spec_nominal")).alias("deviation_from_nominal"),
    ])
    return enriched


def compute_parameter_results(
    df: pl.DataFrame,
    specs: dict,
) -> list[ParameterValidationResult]:
    """
    Compute overall validation summary per parameter.
    """
    enriched = validate_dataframe(df, specs)
    results = []

    for param_name, s in specs.items():
        sub = enriched.filter(pl.col("parameter_name") == param_name)
        if sub.is_empty():
            continue

        n_total = len(sub)
        n_pass  = int(sub.filter(pl.col("pass_fail") == "PASS").height)
        n_fail  = n_total - n_pass
        pass_rate = n_pass / n_total if n_total > 0 else 1.0
        vals = sub["measured_value"].drop_nulls().to_numpy()

        # Status thresholds
        if pass_rate >= 0.99:
            status = "PASS"
        elif pass_rate >= 0.90:
            status = "WARNING"
        else:
            status = "FAIL"

        results.append(ParameterValidationResult(
            parameter=param_name,
            unit=s["unit"],
            lsl=float(s["lsl"]),
            usl=float(s["usl"]),
            nominal=float(s["nominal"]),
            n_total=n_total,
            n_pass=n_pass,
            n_fail=n_fail,
            pass_rate=round(pass_rate, 4),
            fail_rate=round(1.0 - pass_rate, 4),
            min_measured=float(np.min(vals)) if len(vals) else float("nan"),
            max_measured=float(np.max(vals)) if len(vals) else float("nan"),
            mean_measured=float(np.mean(vals)) if len(vals) else float("nan"),
            std_measured=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            status=status,
        ))

    return results


def compute_lot_parameter_matrix(
    df: pl.DataFrame,
    specs: dict,
) -> pl.DataFrame:
    """
    Returns a lot × parameter pass-rate matrix as a Polars DataFrame.
    """
    enriched = validate_dataframe(df, specs)
    rows = []

    for param_name in specs.keys():
        for lot in sorted(df["lot_id"].unique().to_list()):
            sub = enriched.filter(
                (pl.col("parameter_name") == param_name) &
                (pl.col("lot_id") == lot)
            )
            if sub.is_empty():
                continue
            n_total = len(sub)
            n_pass  = int(sub.filter(pl.col("pass_fail") == "PASS").height)
            pass_rate = n_pass / n_total if n_total > 0 else 1.0

            if pass_rate >= 0.99:
                status = "PASS"
            elif pass_rate >= 0.90:
                status = "WARNING"
            else:
                status = "FAIL"

            rows.append({
                "lot_id": lot,
                "parameter_name": param_name,
                "n_total": n_total,
                "n_pass": n_pass,
                "n_fail": n_total - n_pass,
                "pass_rate": round(pass_rate * 100, 2),
                "status": status,
            })

    return pl.DataFrame(rows)


def generate_validation_report_csv(
    df: pl.DataFrame,
    specs: dict,
) -> str:
    """Return validation report as a CSV string."""
    matrix = compute_lot_parameter_matrix(df, specs)
    return matrix.write_csv()
