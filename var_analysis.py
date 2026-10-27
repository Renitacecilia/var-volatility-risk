"""Value-at-Risk (VaR) and volatility analysis for a bank-stock portfolio.

Compares three one-day VaR / Expected Shortfall models (Historical, Parametric
Normal, GARCH(1,1)) and backtests them with the Kupiec and Christoffersen
tests and the Basel traffic-light zones.

Usage:
    python var_analysis.py
    python var_analysis.py --tickers JPM GS BAC MSFT AAPL --start 2015-01-01

Convention: VaR and ES are POSITIVE numbers representing a loss. VaR_99 = 0.025
means "with 99% confidence we will not lose more than 2.5% in one day".
"""
import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

warnings.filterwarnings("ignore", message=".*inplace.*")  # hides noisy yfinance/pandas deprecation warnings

RESULTS = Path("results")
DEFAULT_TICKERS = ["JPM", "GS", "BAC", "MSFT", "AAPL"]


# =============================================================================
# 1. DATA
# =============================================================================

def download_prices(tickers, start="2015-01-01", end=None, cache_path="prices.csv"):
    """Adjusted close prices, one column per ticker. Cached to disk after the first run."""
    if cache_path and Path(cache_path).exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not cached.empty and set(tickers) <= set(cached.columns):
            return cached[list(tickers)]
        print("Cached prices.csv is empty or has different tickers - downloading again.")

    import yfinance as yf

    # threads=False avoids the 'database is locked' errors from parallel downloads
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False, threads=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise RuntimeError(f"No price data downloaded for: {missing}. "
                           "Check your internet connection and the ticker symbols, then re-run.")

    prices = prices[list(tickers)].dropna(how="any")
    if prices.empty:
        raise RuntimeError("Downloaded prices were empty after cleaning. Re-run the script.")
    if cache_path:
        prices.to_csv(cache_path)
    return prices


def portfolio_returns(prices, weights=None):
    """Daily simple returns of a fixed-weight portfolio (equal weights by default)."""
    asset_returns = prices.pct_change().dropna()
    if weights is None:
        w = pd.Series(1 / prices.shape[1], index=prices.columns)
    else:
        w = pd.Series(weights).reindex(prices.columns)
        w = w / w.sum()
    return (asset_returns * w).sum(axis=1).rename("portfolio")


# =============================================================================
# 2. RISK MODELS
# =============================================================================

def historical_var_es(returns, conf=0.99):
    """Non-parametric: use the empirical distribution of past returns."""
    r = np.asarray(returns)
    q = np.quantile(r, 1 - conf)
    return -q, -r[r <= q].mean()


def parametric_var_es(returns, conf=0.99):
    """Variance-covariance method: assume returns are normally distributed."""
    r = np.asarray(returns)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - conf)
    return -(mu + sigma * z), -(mu - sigma * stats.norm.pdf(z) / (1 - conf))


def rolling_var_es(returns, method="historical", window=500, conf=0.99):
    """Forecast for day t uses ONLY the previous `window` days (no look-ahead bias)."""
    func = {"historical": historical_var_es, "parametric": parametric_var_es}[method]
    values = returns.values
    rows = {returns.index[i]: func(values[i - window:i], conf) for i in range(window, len(values))}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["VaR", "ES"])


def garch_recursion(sigma2, resid, omega, alpha, beta):
    """One GARCH(1,1) step: tomorrow's variance from today's variance and shock."""
    return omega + alpha * resid ** 2 + beta * sigma2


def rolling_garch_var_es(returns, window=500, conf=0.99, refit_every=21):
    """GARCH(1,1) volatility forecasts turned into VaR/ES (normal innovations).

    Re-fitted every `refit_every` days on the previous `window` days; between
    refits the variance is updated daily with the fitted parameters.
    """
    from arch import arch_model

    r = returns * 100  # arch fits better with returns in percent
    z = stats.norm.ppf(1 - conf)
    pdf_term = stats.norm.pdf(z) / (1 - conf)
    rows = {}

    for start in range(window, len(r), refit_every):
        fit = arch_model(r.iloc[start - window:start], mean="Constant", vol="GARCH",
                         p=1, q=1, dist="normal").fit(disp="off")
        mu = fit.params["mu"]
        omega, a, b = fit.params["omega"], fit.params["alpha[1]"], fit.params["beta[1]"]
        sigma2 = float(fit.forecast(horizon=1).variance.iloc[-1, 0])

        for i in range(start, min(start + refit_every, len(r))):
            sigma = np.sqrt(sigma2)
            rows[r.index[i]] = (-(mu + sigma * z) / 100, -(mu - sigma * pdf_term) / 100)
            sigma2 = garch_recursion(sigma2, r.iloc[i] - mu, omega, a, b)

    return pd.DataFrame.from_dict(rows, orient="index", columns=["VaR", "ES"])


# =============================================================================
# 3. BACKTESTING
# =============================================================================

def exceedances(returns, var):
    """1 on days when the actual loss was bigger than the VaR forecast."""
    return (returns.loc[var.index] < -var).astype(int)


def kupiec_pof(hits, conf):
    """Kupiec test: is the exceedance RATE right? p < 0.05 rejects the model."""
    n, x = len(hits), int(hits.sum())
    p, phat = 1 - conf, x / n
    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_independence(hits):
    """Christoffersen test: are exceedances CLUSTERED? p < 0.05 means yes (bad)."""
    h = np.asarray(hits)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0))
    n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0))
    n11 = np.sum((prev == 1) & (curr == 1))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def basel_zone(exceptions_last_250):
    """Basel traffic-light zone for a 99% VaR over the last 250 trading days."""
    if exceptions_last_250 <= 4:
        return "green"
    return "yellow" if exceptions_last_250 <= 9 else "red"


def backtest_summary(returns, var_df, conf):
    hits = exceedances(returns, var_df["VaR"])
    n, x = len(hits), int(hits.sum())
    _, p_pof = kupiec_pof(hits, conf)
    _, p_ind = christoffersen_independence(hits)
    return {
        "observations": n,
        "expected_exceedances": round(n * (1 - conf), 1),
        "actual_exceedances": x,
        "exceedance_rate": round(x / n, 4),
        "kupiec_p_value": round(p_pof, 4),
        "christoffersen_p_value": round(p_ind, 4),
        "basel_zone_last_250": basel_zone(int(hits.iloc[-250:].sum())),
        "avg_VaR": round(var_df["VaR"].mean(), 4),
    }


# =============================================================================
# 4. PLOTS
# =============================================================================

def plot_returns_and_vol(returns, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(returns.index, returns, lw=0.6)
    axes[0].set_title("Daily portfolio returns")
    axes[1].plot(returns.index, returns.rolling(21).std() * (252 ** 0.5), lw=1)
    axes[1].set_title("Rolling 21-day annualised volatility (note the volatility clustering)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_var(returns, var_df, name, conf, path):
    hits = exceedances(returns, var_df["VaR"])
    aligned = returns.loc[var_df.index]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(aligned.index, aligned, lw=0.5, color="grey", label="Daily return")
    ax.plot(var_df.index, -var_df["VaR"], color="crimson", lw=1, label=f"-VaR {conf:.0%}")
    ax.scatter(aligned.index[hits == 1], aligned[hits == 1], color="black", s=14, zorder=3,
               label=f"Exceedances ({int(hits.sum())})")
    ax.set_title(f"{name}: {conf:.0%} one-day VaR vs actual returns")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 5. MAIN
# =============================================================================

def build_forecasts(returns, conf, window):
    forecasts = {
        "Historical": rolling_var_es(returns, "historical", window, conf),
        "Parametric (Normal)": rolling_var_es(returns, "parametric", window, conf),
    }
    try:
        forecasts["GARCH(1,1)"] = rolling_garch_var_es(returns, window, conf)
    except ImportError:
        print("The 'arch' package is not installed - skipping GARCH. Run: pip install arch")
    return forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    prices = download_prices(args.tickers, start=args.start)
    returns = portfolio_returns(prices)
    print(f"{len(returns)} daily returns from {returns.index[0].date()} to {returns.index[-1].date()}")

    plot_returns_and_vol(returns, RESULTS / "returns_and_volatility.png")

    rows = []
    for conf in (0.95, 0.99):
        for name, var_df in build_forecasts(returns, conf, args.window).items():
            row = backtest_summary(returns, var_df, conf)
            row.update(model=name, confidence=conf, avg_ES=round(var_df["ES"].mean(), 4))
            rows.append(row)
            if conf == 0.99:
                tag = name.split()[0].lower().replace("(", "").replace(")", "").replace(",", "")
                plot_var(returns, var_df, name, conf, RESULTS / f"var99_{tag}.png")

    summary = pd.DataFrame(rows)
    summary = summary[["model", "confidence"] + [c for c in summary.columns if c not in ("model", "confidence")]]
    summary.to_csv(RESULTS / "backtest_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
