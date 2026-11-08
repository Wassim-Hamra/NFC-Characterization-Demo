"""
data_generator.py
-----------------
Generates a reproducible synthetic NFC IC characterization dataset.

DISCLAIMER: All chip IDs, lot IDs, measurement values, and specification limits
are entirely synthetic. This is an independent educational prototype not affiliated
with NXP Semiconductors.
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

RANDOM_SEED = 42
N_RECORDS = 5000

LOTS = {
    "LOT-A": {"scenario": "normal",    "description": "Normal production lot — reference baseline"},
    "LOT-B": {"scenario": "normal",    "description": "Normal production lot — slight voltage corner variation"},
    "LOT-C": {"scenario": "shifted",   "description": "Systematic shift in receiver sensitivity (–3 dBm offset)"},
    "LOT-D": {"scenario": "bimodal",   "description": "Bimodal load modulation distribution"},
    "LOT-E": {"scenario": "normal",    "description": "Normal production lot — slight spread increase"},
}

PARAMETERS = {
    "carrier_frequency_mhz": {
        "unit": "MHz", "nominal": 13.56, "sigma": 0.015,
        "lsl": 13.50, "usl": 13.62,
    },
    "load_modulation_mv": {
        "unit": "mV", "nominal": 160.0, "sigma": 8.0,
        "lsl": 120.0, "usl": 200.0,
    },
    "receiver_sensitivity_dbm": {
        "unit": "dBm", "nominal": -60.0, "sigma": 1.5,
        "lsl": -65.0, "usl": -50.0,
    },
    "modulation_depth_pct": {
        "unit": "%", "nominal": 10.0, "sigma": 0.6,
        "lsl": 7.0, "usl": 14.0,
    },
}

TEMPERATURES = [0.0, 25.0, 85.0]
VOLTAGES = [3.0, 3.3, 3.6]
TEST_CONDITIONS = {
    (0.0, 3.0): "COLD_LOW_VDD",
    (25.0, 3.3): "NOMINAL",
    (85.0, 3.3): "HOT",
    (25.0, 3.0): "LOW_VDD",
    (25.0, 3.6): "HIGH_VDD",
    (85.0, 3.0): "HOT_LOW_VDD",
}
TEST_SITES = ["SITE-1", "SITE-2", "SITE-3"]


def _chip_ids_for_lot(lot_id: str, n_chips: int, rng: np.random.Generator) -> list[str]:
    return [f"CHIP-{lot_id}-{i:04d}" for i in rng.choice(n_chips * 3, n_chips, replace=False)]


def _wafer_ids(n: int) -> list[str]:
    return [f"W{i:02d}" for i in range(1, n + 1)]


def _generate_lot_records(
    lot_id: str,
    scenario: str,
    n_per_lot: int,
    rng: np.random.Generator,
    start_time: datetime,
) -> list[dict]:

    n_chips = max(20, n_per_lot // 10)
    chip_ids = _chip_ids_for_lot(lot_id, n_chips, rng)
    wafer_ids = _wafer_ids(5)
    records = []

    for i in range(n_per_lot):
        chip_id = chip_ids[i % len(chip_ids)]
        wafer_id = wafer_ids[i % len(wafer_ids)]
        test_id = f"T-{lot_id}-{i:05d}"

        # Pick a temperature/voltage condition
        cond_pair = list(TEST_CONDITIONS.keys())[i % len(TEST_CONDITIONS)]
        temperature_c, supply_voltage_v = cond_pair
        test_condition = TEST_CONDITIONS[cond_pair]
        test_site = TEST_SITES[i % len(TEST_SITES)]
        test_timestamp = start_time + timedelta(hours=i * 0.1)

        # Measure all 4 parameters for this chip/condition
        for param_name, pspec in PARAMETERS.items():
            nominal = pspec["nominal"]
            sigma = pspec["sigma"]

            # Temperature coefficient (illustrative)
            temp_coeff = 0.0
            if param_name == "carrier_frequency_mhz":
                temp_coeff = -0.0001 * (temperature_c - 25.0)
            elif param_name == "receiver_sensitivity_dbm":
                temp_coeff = 0.015 * (temperature_c - 25.0)
            elif param_name == "modulation_depth_pct":
                temp_coeff = -0.005 * (temperature_c - 25.0)

            # Voltage coefficient (illustrative)
            volt_coeff = 0.0
            if param_name == "load_modulation_mv":
                volt_coeff = 5.0 * (supply_voltage_v - 3.3)
            elif param_name == "receiver_sensitivity_dbm":
                volt_coeff = -0.5 * (supply_voltage_v - 3.3)

            base = nominal + temp_coeff + volt_coeff
            value = rng.normal(base, sigma)

            # ── Scenario overrides ──────────────────────────────
            if scenario == "shifted" and param_name == "receiver_sensitivity_dbm":
                # LOT-C: systematic degraded sensitivity (+8.5 dBm shift towards and across USL)
                value = rng.normal(base + 8.5, sigma)

            elif scenario == "bimodal" and param_name == "load_modulation_mv":
                # LOT-D: two populations (mu1=145, mu2=185)
                if rng.random() < 0.45:
                    value = rng.normal(base - 15.0, sigma * 1.2)   # low population
                else:
                    value = rng.normal(base + 25.0, sigma * 1.2)   # high population

            elif scenario == "normal" and lot_id == "LOT-B":
                # LOT-B: slight supply-voltage corner sensitivity
                volt_extra = 0.8 * (supply_voltage_v - 3.3)
                if param_name == "carrier_frequency_mhz":
                    value = rng.normal(base + volt_extra * 0.002, sigma)

            elif scenario == "normal" and lot_id == "LOT-E":
                # LOT-E: slightly wider spread
                value = rng.normal(base, sigma * 1.3)

            records.append({
                "chip_id": chip_id,
                "lot_id": lot_id,
                "wafer_id": wafer_id,
                "test_id": test_id,
                "parameter_name": param_name,
                "measured_value": round(float(value), 5),
                "unit": pspec["unit"],
                "temperature_c": temperature_c,
                "supply_voltage_v": supply_voltage_v,
                "test_site": test_site,
                "test_timestamp": test_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "test_condition": test_condition,
            })

    return records


def generate_dataset(n_total: int = N_RECORDS) -> pl.DataFrame:
    """Generate and return the full synthetic dataset as a Polars DataFrame."""
    rng = np.random.default_rng(RANDOM_SEED)
    all_records = []
    # Each test run records all parameters in PARAMETERS
    n_per_lot = n_total // (len(LOTS) * len(PARAMETERS))

    base_time = datetime(2024, 3, 1, 8, 0, 0)
    lot_offsets = [timedelta(days=d) for d in [0, 15, 30, 45, 60]]

    for (lot_id, lot_info), offset in zip(LOTS.items(), lot_offsets):
        records = _generate_lot_records(
            lot_id=lot_id,
            scenario=lot_info["scenario"],
            n_per_lot=n_per_lot,
            rng=rng,
            start_time=base_time + offset,
        )
        all_records.extend(records)

    df = pl.DataFrame(all_records)
    # Cast timestamp properly
    df = df.with_columns(
        pl.col("test_timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S")
    )
    return df


def save_dataset(df: pl.DataFrame, path: str) -> None:
    """Save the dataset to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def load_or_generate(csv_path: str = "data/synthetic_measurements.csv") -> pl.DataFrame:
    """Load from CSV if it exists, otherwise generate and save."""
    if os.path.exists(csv_path):
        df = pl.read_csv(csv_path, try_parse_dates=True)
        return df
    df = generate_dataset()
    save_dataset(df, csv_path)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df, "data/synthetic_measurements.csv")
    print(f"Generated {len(df)} records across {df['lot_id'].n_unique()} lots.")
    print(df.head(5))
