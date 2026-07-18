# Value-at-Risk Modelling for SPY

This project estimates the 30-trading-day 99% Value-at-Risk (VaR) of a $10 million long position in the SPDR S&P 500 ETF (SPY).

The analysis implements three standard VaR methodologies:

- Parametric VaR
- Historical Simulation VaR
- Monte Carlo VaR

The project was developed as part of a Computational Finance module in the MSc Finance & Analytics programme at King's College London.

## Project Objective

The aim of this project is to compare how different VaR modelling assumptions affect estimated downside risk.

The analysis estimates VaR as of two valuation dates:

- the last trading day in November 2024
- the last trading day in November 2025

Only information available up to each valuation date is used, avoiding look-ahead bias.

## Methodology

Daily close-to-close logarithmic returns are calculated from SPY closing prices.

The 30-day 99% VaR is then estimated using three approaches.

### Parametric VaR

The parametric approach assumes daily log returns are independently and normally distributed.

The daily mean and volatility are estimated from the historical sample and scaled to a 30-trading-day horizon. The 1% lower-tail normal quantile is then converted into a dollar loss on a $10 million position.

### Historical Simulation VaR

The historical simulation approach does not impose a normal distribution.

Instead, overlapping 30-day cumulative log returns are constructed directly from historical SPY returns. These are converted into dollar profit and loss values, and the 1st percentile of the empirical distribution is used to estimate 99% VaR.

### Monte Carlo VaR

The Monte Carlo approach simulates future daily log returns using the estimated historical mean and volatility.

The simulated returns are aggregated over 30 trading days, converted into dollar profit and loss outcomes, and the 1st percentile of the simulated distribution is used to estimate VaR.

## Additional Analysis

The project also includes:

- Expected Shortfall calculations
- estimation-window sensitivity analysis
- comparison of 1-year, 3-year, 5-year, and full-sample VaR estimates
- discussion of modelling assumptions and limitations

## Key Findings

The historical simulation method produces higher VaR estimates than the parametric and Monte Carlo approaches.

This suggests that the empirical downside tail of SPY returns is heavier than the Gaussian approximation used in the parametric and Monte Carlo models.

The parametric and Monte Carlo VaR estimates are very similar because both methods rely on the same normal/lognormal return assumptions.

The sensitivity analysis shows that VaR estimates can change materially depending on the estimation window used, highlighting the importance of volatility regimes and sample selection.

## Files

- `var_analysis.py` — main Python script for VaR and Expected Shortfall estimation
- `requirements.txt` — Python package requirements
- `DataVaR.xlsx` — input dataset, if included locally

## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
