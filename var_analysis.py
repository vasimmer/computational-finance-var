"""Value-at-Risk analysis for a USD 10 million long SPY position.

The script estimates 30-trading-day 99% VaR at the last available trading
session in November 2024 and November 2025 using:

1. Parametric (variance-covariance) VaR
2. Historical simulation
3. Monte Carlo simulation under iid Gaussian log returns

It also calculates Expected Shortfall and parametric VaR sensitivity using
1-year, 3-year, 5-year, and full expanding estimation windows.

Expected input
--------------
An Excel workbook named ``DataVaR.xlsx`` containing at least:
- Date
- Close

Run
---
    python var_analysis.py

Install dependencies
--------------------
    pip install pandas numpy scipy openpyxl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

DATA_FILE = Path("DataVaR.xlsx")
POSITION_VALUE = 10_000_000.0
HORIZON_DAYS = 30
CONFIDENCE_LEVEL = 0.99
ALPHA = 1.0 - CONFIDENCE_LEVEL
N_SIMULATIONS = 100_000
RANDOM_SEED = 42
TARGET_MONTHS = ((2024, 11), (2025, 11))
SENSITIVITY_YEARS = (1, 3, 5)
RESULTS_FILE = Path("var_results.csv")


@dataclass(frozen=True)
class RiskEstimate:
    """Container for a VaR and Expected Shortfall estimate."""

    var: float
    expected_shortfall: float | None = None


# -----------------------------------------------------------------------------
# DATA LOADING AND CLEANING
# -----------------------------------------------------------------------------


def _find_column(columns: Iterable[str], required_name: str) -> str:
    """Find a column case-insensitively and ignore surrounding whitespace."""
    normalised = {str(column).strip().lower(): str(column) for column in columns}
    key = required_name.strip().lower()
    if key not in normalised:
        raise KeyError(
            f"Required column '{required_name}' was not found. "
            f"Available columns: {list(columns)}"
        )
    return normalised[key]


def load_spy_data(file_path: Path) -> pd.DataFrame:
    """Load, validate, clean, and sort the SPY price data."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Place DataVaR.xlsx in the same "
            "folder as this script or change DATA_FILE."
        )

    raw = pd.read_excel(file_path)
    date_column = _find_column(raw.columns, "Date")
    close_column = _find_column(raw.columns, "Close")

    data = raw[[date_column, close_column]].copy()
    data.columns = ["Date", "Close"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    invalid_rows = data["Date"].isna() | data["Close"].isna()
    if invalid_rows.any():
        print(f"Warning: dropping {int(invalid_rows.sum())} invalid row(s).")
        data = data.loc[~invalid_rows].copy()

    if (data["Close"] <= 0).any():
        raise ValueError("All closing prices must be strictly positive.")

    duplicate_count = int(data["Date"].duplicated().sum())
    if duplicate_count:
        print(
            f"Warning: found {duplicate_count} duplicate date(s); "
            "keeping the final observation for each date."
        )
        data = data.drop_duplicates(subset="Date", keep="last")

    data = data.sort_values("Date").reset_index(drop=True)

    # Compute close-to-close daily log returns once on the complete dataset.
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    if data.empty or data["LogReturn"].dropna().empty:
        raise ValueError("The workbook does not contain enough valid price data.")

    return data


def last_trading_day_in_month(data: pd.DataFrame, year: int, month: int) -> pd.Timestamp:
    """Return the final available trading date in a specified calendar month."""
    mask = (data["Date"].dt.year == year) & (data["Date"].dt.month == month)
    dates = data.loc[mask, "Date"]
    if dates.empty:
        raise ValueError(f"No observations found for {year}-{month:02d}.")
    return pd.Timestamp(dates.max())


def returns_up_to(data: pd.DataFrame, valuation_date: pd.Timestamp) -> pd.Series:
    """Return all non-missing log returns available by the valuation date."""
    returns = data.loc[data["Date"] <= valuation_date, "LogReturn"].dropna()
    if len(returns) < HORIZON_DAYS:
        raise ValueError(
            f"Only {len(returns)} returns are available by {valuation_date.date()}; "
            f"at least {HORIZON_DAYS} are required."
        )
    return returns.astype(float)


# -----------------------------------------------------------------------------
# RISK-MEASURE FUNCTIONS
# -----------------------------------------------------------------------------


def loss_from_log_return(log_return: np.ndarray | float, position: float) -> np.ndarray | float:
    """Convert a log return into a positive loss amount.

    Portfolio P/L is position * (exp(r) - 1). Therefore, positive loss is the
    negative of P/L: position * (1 - exp(r)).
    """
    return position * (1.0 - np.exp(log_return))


def parametric_var(
    log_returns: pd.Series,
    position: float = POSITION_VALUE,
    horizon_days: int = HORIZON_DAYS,
    alpha: float = ALPHA,
) -> RiskEstimate:
    """Calculate Gaussian parametric VaR and ES from daily log returns."""
    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std(ddof=1))

    mu_horizon = horizon_days * mu_daily
    sigma_horizon = np.sqrt(horizon_days) * sigma_daily

    z_alpha = float(norm.ppf(alpha))
    quantile_log_return = mu_horizon + sigma_horizon * z_alpha
    var_value = float(loss_from_log_return(quantile_log_return, position))

    # Closed-form Gaussian ES in log-return space is not transformed linearly
    # into dollar P/L. We therefore leave parametric ES unreported here rather
    # than present an inaccurate approximation.
    return RiskEstimate(var=max(var_value, 0.0))


def historical_var_es(
    log_returns: pd.Series,
    position: float = POSITION_VALUE,
    horizon_days: int = HORIZON_DAYS,
    alpha: float = ALPHA,
) -> RiskEstimate:
    """Calculate historical VaR and ES from overlapping horizon returns."""
    horizon_log_returns = log_returns.rolling(horizon_days).sum().dropna()
    pnl = position * (np.exp(horizon_log_returns.to_numpy()) - 1.0)

    pnl_quantile = float(np.quantile(pnl, alpha))
    var_value = max(-pnl_quantile, 0.0)

    tail_losses = -pnl[pnl <= pnl_quantile]
    es_value = float(tail_losses.mean())

    return RiskEstimate(var=var_value, expected_shortfall=es_value)


def monte_carlo_var_es(
    log_returns: pd.Series,
    position: float = POSITION_VALUE,
    horizon_days: int = HORIZON_DAYS,
    alpha: float = ALPHA,
    n_simulations: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> RiskEstimate:
    """Calculate Monte Carlo VaR and ES using iid Gaussian daily log returns."""
    if n_simulations < 10_000:
        print("Warning: fewer than 10,000 simulations may give unstable tail estimates.")

    mu_daily = float(log_returns.mean())
    sigma_daily = float(log_returns.std(ddof=1))

    rng = np.random.default_rng(seed)
    simulated_daily_returns = rng.normal(
        loc=mu_daily,
        scale=sigma_daily,
        size=(n_simulations, horizon_days),
    )
    simulated_horizon_returns = simulated_daily_returns.sum(axis=1)
    pnl = position * (np.exp(simulated_horizon_returns) - 1.0)

    pnl_quantile = float(np.quantile(pnl, alpha))
    var_value = max(-pnl_quantile, 0.0)

    tail_losses = -pnl[pnl <= pnl_quantile]
    es_value = float(tail_losses.mean())

    return RiskEstimate(var=var_value, expected_shortfall=es_value)


# -----------------------------------------------------------------------------
# ESTIMATION-WINDOW SENSITIVITY
# -----------------------------------------------------------------------------


def window_returns(
    data: pd.DataFrame,
    valuation_date: pd.Timestamp,
    years: int | None,
) -> pd.Series:
    """Return an expanding or trailing calendar-year estimation sample."""
    mask = data["Date"] <= valuation_date

    if years is not None:
        start_date = valuation_date - pd.DateOffset(years=years)
        mask &= data["Date"] > start_date

    sample = data.loc[mask, "LogReturn"].dropna().astype(float)
    if len(sample) < HORIZON_DAYS:
        raise ValueError(
            f"The {years}-year window ending {valuation_date.date()} contains "
            f"only {len(sample)} valid returns."
        )
    return sample


def parametric_window_sensitivity(
    data: pd.DataFrame,
    valuation_date: pd.Timestamp,
    years_list: tuple[int, ...] = SENSITIVITY_YEARS,
) -> dict[str, float]:
    """Calculate parametric VaR for trailing windows and the full sample."""
    estimates: dict[str, float] = {}

    for years in years_list:
        sample = window_returns(data, valuation_date, years)
        estimates[f"{years}-year"] = parametric_var(sample).var

    full_sample = window_returns(data, valuation_date, None)
    estimates["Full sample"] = parametric_var(full_sample).var

    return estimates


# -----------------------------------------------------------------------------
# MAIN ANALYSIS
# -----------------------------------------------------------------------------


def analyse_valuation_date(
    data: pd.DataFrame,
    valuation_date: pd.Timestamp,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run all models for one valuation date and return tidy result rows."""
    log_returns = returns_up_to(data, valuation_date)

    parametric = parametric_var(log_returns)
    historical = historical_var_es(log_returns)
    monte_carlo = monte_carlo_var_es(log_returns, seed=seed)

    model_rows = [
        {
            "Valuation Date": valuation_date.date().isoformat(),
            "Method": "Parametric",
            "VaR": parametric.var,
            "Expected Shortfall": np.nan,
            "Observations": len(log_returns),
        },
        {
            "Valuation Date": valuation_date.date().isoformat(),
            "Method": "Historical simulation",
            "VaR": historical.var,
            "Expected Shortfall": historical.expected_shortfall,
            "Observations": len(log_returns),
        },
        {
            "Valuation Date": valuation_date.date().isoformat(),
            "Method": "Monte Carlo",
            "VaR": monte_carlo.var,
            "Expected Shortfall": monte_carlo.expected_shortfall,
            "Observations": len(log_returns),
        },
    ]

    sensitivity = parametric_window_sensitivity(data, valuation_date)
    sensitivity_rows = [
        {
            "Valuation Date": valuation_date.date().isoformat(),
            "Window": window,
            "Parametric VaR": value,
        }
        for window, value in sensitivity.items()
    ]

    return model_rows, sensitivity_rows


def print_currency_table(frame: pd.DataFrame, currency_columns: list[str]) -> None:
    """Print a DataFrame with readable US-dollar formatting."""
    printable = frame.copy()
    for column in currency_columns:
        printable[column] = printable[column].map(
            lambda value: "" if pd.isna(value) else f"${value:,.2f}"
        )
    print(printable.to_string(index=False))


def main() -> None:
    data = load_spy_data(DATA_FILE)

    valuation_dates = [
        last_trading_day_in_month(data, year, month)
        for year, month in TARGET_MONTHS
    ]

    all_model_rows: list[dict[str, object]] = []
    all_sensitivity_rows: list[dict[str, object]] = []

    for index, valuation_date in enumerate(valuation_dates):
        model_rows, sensitivity_rows = analyse_valuation_date(
            data=data,
            valuation_date=valuation_date,
            seed=RANDOM_SEED + index,
        )
        all_model_rows.extend(model_rows)
        all_sensitivity_rows.extend(sensitivity_rows)

    model_results = pd.DataFrame(all_model_rows)
    sensitivity_results = pd.DataFrame(all_sensitivity_rows)

    print("\n30-TRADING-DAY 99% VALUE-AT-RISK AND EXPECTED SHORTFALL\n")
    print_currency_table(model_results, ["VaR", "Expected Shortfall"])

    print("\nPARAMETRIC VaR: ESTIMATION-WINDOW SENSITIVITY\n")
    print_currency_table(sensitivity_results, ["Parametric VaR"])

    combined_results = model_results.copy()
    combined_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nMain results saved to: {RESULTS_FILE.resolve()}")


if __name__ == "__main__":
    main()
