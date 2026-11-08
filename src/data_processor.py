"""
data_processor.py
-----------------
Polars-based data preparation and cleaning pipeline.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field

EXPECTED_COLUMNS = [
    "chip_id", "lot_id", "wafer_id", "test_id",
    "parameter_name", "measured_value", "unit",
    "temperature_c", "supply_voltage_v",
    "test_site", "test_timestamp", "test_condition",
]

NUMERIC_COLUMNS = ["measured_value", "temperature_c", "supply_voltage_v"]
STRING_COLUMNS  = ["chip_id", "lot_id", "wafer_id", "test_id",
                   "parameter_name", "unit", "test_site", "test_condition"]


@dataclass
class ProcessingReport:
    n_input: int = 0
    n_output: int = 0
    missing_columns: list[str] = field(default_factory=list)
    n_duplicates_removed: int = 0
    n_null_measured_value: int = 0
    n_null_filled: int = 0
    n_inf_flagged: int = 0
    n_type_cast_errors: int = 0
    n_out_of_range_flagged: int = 0
    warnings: list[str] = field(default_factory=list)
    columns_renamed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "Records in": self.n_input,
            "Records out": self.n_output,
            "Missing columns": ", ".join(self.missing_columns) if self.missing_columns else "None",
            "Duplicates removed": self.n_duplicates_removed,
            "Null measured_value rows": self.n_null_measured_value,
            "Inf / NaN flagged": self.n_inf_flagged,
            "Out-of-range flagged": self.n_out_of_range_flagged,
            "Warnings": "; ".join(self.warnings) if self.warnings else "None",
        }


def _standardise_column_names(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    """Lowercase column names, replace spaces with underscores."""
    rename_map = {}
    for col in df.columns:
        new_name = col.strip().lower().replace(" ", "_").replace("-", "_")
        if new_name != col:
            rename_map[col] = new_name
    if rename_map:
        df = df.rename(rename_map)
        report.columns_renamed = rename_map
    return df


def _check_expected_columns(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report.missing_columns = missing
    if missing:
        report.warnings.append(f"Missing expected columns: {missing}")
    return df


def _cast_types(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    """Ensure numeric columns are Float64."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except Exception:
                report.n_type_cast_errors += 1
                report.warnings.append(f"Could not cast column '{col}' to Float64")
    return df


def _remove_duplicates(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    n_before = len(df)
    df = df.unique()
    report.n_duplicates_removed = n_before - len(df)
    return df


def _handle_nulls(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    if "measured_value" not in df.columns:
        return df
    null_mask = df["measured_value"].is_null()
    report.n_null_measured_value = int(null_mask.sum())
    if report.n_null_measured_value > 0:
        # Flag nulls with a column, then drop them
        df = df.with_columns(
            pl.when(pl.col("measured_value").is_null())
            .then(pl.lit("NULL_MEASURED_VALUE"))
            .otherwise(pl.lit(None))
            .alias("data_quality_flag")
        )
        df = df.filter(pl.col("measured_value").is_not_null())
        report.warnings.append(
            f"{report.n_null_measured_value} rows with null measured_value were removed."
        )
    return df


def _flag_infinite(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    if "measured_value" not in df.columns:
        return df
    inf_mask = df["measured_value"].is_infinite()
    report.n_inf_flagged = int(inf_mask.sum())
    if report.n_inf_flagged > 0:
        df = df.filter(~pl.col("measured_value").is_infinite())
        report.warnings.append(
            f"{report.n_inf_flagged} rows with infinite measured_value were removed."
        )
    return df


def _flag_extreme_outliers(df: pl.DataFrame, report: ProcessingReport) -> pl.DataFrame:
    """Flag values beyond ±10σ from their parameter mean as extreme outliers."""
    if "measured_value" not in df.columns or "parameter_name" not in df.columns:
        return df

    # Compute per-parameter stats
    stats = (
        df.group_by("parameter_name")
        .agg([
            pl.col("measured_value").mean().alias("param_mean"),
            pl.col("measured_value").std().alias("param_std"),
        ])
    )
    df = df.join(stats, on="parameter_name", how="left")
    outlier_mask = (
        (pl.col("measured_value") - pl.col("param_mean")).abs()
        > 10.0 * pl.col("param_std")
    )
    n_outliers = int(df.filter(outlier_mask).height)
    report.n_out_of_range_flagged = n_outliers
    if n_outliers > 0:
        report.warnings.append(
            f"{n_outliers} extreme outliers (>10σ) flagged and removed."
        )
        df = df.filter(~outlier_mask)

    df = df.drop(["param_mean", "param_std"])
    return df


def process(df: pl.DataFrame) -> tuple[pl.DataFrame, ProcessingReport]:
    """
    Run the full data preparation pipeline.

    Returns:
        (cleaned_df, report)
    """
    report = ProcessingReport(n_input=len(df))

    df = _standardise_column_names(df, report)
    df = _check_expected_columns(df, report)
    df = _cast_types(df, report)
    df = _remove_duplicates(df, report)
    df = _handle_nulls(df, report)
    df = _flag_infinite(df, report)
    df = _flag_extreme_outliers(df, report)

    report.n_output = len(df)
    return df, report


def validate_upload_schema(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """
    Check whether an uploaded DataFrame has the minimum required columns.
    Returns (is_valid, list_of_missing_columns).
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing
