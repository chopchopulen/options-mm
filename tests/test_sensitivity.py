"""
Tests for src/backtest/sensitivity.py
"""

import os
import math
import pandas as pd
import pytest

from src.backtest.sensitivity import run_sensitivity

# Canonical small grid for fast tests: single combo
SMALL_GRID = [(25, 20, 0.002)]


class TestSensitivityCsvCreated:
    def test_sensitivity_csv_created(self, tmp_path, monkeypatch):
        """run_sensitivity with a single combo should create results/sensitivity.csv."""
        # Point the working directory to tmp_path so the CSV lands there
        monkeypatch.chdir(tmp_path)

        run_sensitivity(seeds=[42], grid_override=SMALL_GRID)

        csv_path = tmp_path / "results" / "sensitivity.csv"
        assert csv_path.exists(), "results/sensitivity.csv was not created"

        df = pd.read_csv(csv_path)
        expected_columns = {
            "hedge_threshold",
            "base_spread_bps",
            "informed_threshold",
            "mean_sharpe",
            "std_sharpe",
            "mean_total_pnl",
            "mean_hedge_cost",
            "mean_spread_capture",
        }
        assert expected_columns.issubset(set(df.columns)), (
            f"Missing columns: {expected_columns - set(df.columns)}"
        )


class TestSensitivityReturnsDataframe:
    def test_sensitivity_returns_dataframe(self, tmp_path, monkeypatch):
        """run_sensitivity should return a pandas DataFrame with mean_sharpe column."""
        monkeypatch.chdir(tmp_path)

        result = run_sensitivity(seeds=[42], grid_override=SMALL_GRID)

        assert isinstance(result, pd.DataFrame), "Return value should be a DataFrame"
        assert "mean_sharpe" in result.columns, "DataFrame must have a mean_sharpe column"
        assert len(result) == len(SMALL_GRID), (
            f"Expected {len(SMALL_GRID)} rows, got {len(result)}"
        )


class TestSensitivityBestComboIsValid:
    def test_sensitivity_best_combo_is_valid(self, tmp_path, monkeypatch):
        """Best combo on the full grid (2 seeds) should have valid float metrics."""
        monkeypatch.chdir(tmp_path)

        df = run_sensitivity(seeds=[42, 43])

        # Sorted descending by mean_sharpe, so first row is best
        best = df.iloc[0]

        metric_cols = [
            "mean_sharpe",
            "std_sharpe",
            "mean_total_pnl",
            "mean_hedge_cost",
            "mean_spread_capture",
        ]
        for col in metric_cols:
            val = best[col]
            assert isinstance(val, float), f"{col} should be float, got {type(val)}"
            assert math.isfinite(val), f"{col} should be finite, got {val}"
