"""
statistics_engine.py
--------------------
Descriptive statistics for a filtered measurement slice.
"""

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from dataclasses import dataclass


@dataclass
class StatSummary:
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p75: float
    p95: float
    skewness: float
    kurtosis: float
    cv_pct: float          # coefficient of variation %
    cp: float | None       # process capability (if spec provided)
    cpk: float | None      # process capability index (if spec provided)

    def to_dict(self) -> dict:
        d = {
            "Count": self.count,
            "Mean": round(self.mean, 4),
            "Median": round(self.median, 4),
            "Std Dev": round(self.std, 4),
            "Min": round(self.min, 4),
            "Max": round(self.max, 4),
            "5th Pct": round(self.p5, 4),
            "25th Pct": round(self.p25, 4),
            "75th Pct": round(self.p75, 4),
            "95th Pct": round(self.p95, 4),
            "Skewness": round(self.skewness, 4),
            "Kurtosis": round(self.kurtosis, 4),
            "CV (%)": round(self.cv_pct, 2),
        }
        if self.cp is not None:
            d["Cp"] = round(self.cp, 3)
        if self.cpk is not None:
            d["Cpk"] = round(self.cpk, 3)
        return d


def compute_stats(
    values: np.ndarray,
    lsl: float | None = None,
    usl: float | None = None,
) -> StatSummary:
    """Compute descriptive statistics for a 1-D array of measurements."""
    if len(values) == 0:
        raise ValueError("Cannot compute statistics on an empty array.")

    values = values[~np.isnan(values)]
    if len(values) == 0:
        raise ValueError("All values are NaN.")

    mean = float(np.mean(values))
    std  = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv   = (std / abs(mean) * 100.0) if mean != 0 else 0.0

    skewness = float(scipy_stats.skew(values)) if len(values) > 2 else 0.0
    kurtosis = float(scipy_stats.kurtosis(values)) if len(values) > 3 else 0.0

    # Process capability
    cp = cpk = None
    if lsl is not None and usl is not None and std > 0:
        cp  = (usl - lsl) / (6.0 * std)
        cpk = min((usl - mean) / (3.0 * std), (mean - lsl) / (3.0 * std))

    return StatSummary(
        count=len(values),
        mean=mean,
        median=float(np.median(values)),
        std=std,
        min=float(np.min(values)),
        max=float(np.max(values)),
        p5=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        skewness=skewness,
        kurtosis=kurtosis,
        cv_pct=cv,
        cp=cp,
        cpk=cpk,
    )


def compute_stats_from_df(
    df: pl.DataFrame,
    parameter: str,
    lsl: float | None = None,
    usl: float | None = None,
) -> StatSummary:
    """Filter df by parameter, then compute stats."""
    filtered = df.filter(pl.col("parameter_name") == parameter)
    if filtered.is_empty():
        raise ValueError(f"No data for parameter '{parameter}'.")
    values = filtered["measured_value"].to_numpy()
    return compute_stats(values, lsl=lsl, usl=usl)


def per_lot_stats(
    df: pl.DataFrame,
    parameter: str,
    lsl: float | None = None,
    usl: float | None = None,
) -> dict[str, StatSummary]:
    """Compute stats for each lot for the given parameter."""
    result = {}
    lots = df["lot_id"].unique().to_list()
    for lot in sorted(lots):
        sub = df.filter(
            (pl.col("lot_id") == lot) & (pl.col("parameter_name") == parameter)
        )
        if sub.is_empty():
            continue
        values = sub["measured_value"].to_numpy()
        result[lot] = compute_stats(values, lsl=lsl, usl=usl)
    return result
