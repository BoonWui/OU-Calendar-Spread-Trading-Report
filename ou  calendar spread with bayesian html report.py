# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:12:58 2026

@author: BWLAU
"""


# -*- coding: utf-8 -*-
"""
Rolling OU Pairs Trading - Calendar Spread (Price Space)
Bayesian upgrade:
- Replace OLS hedge (alpha,beta) with Bayesian shrinkage regression (conjugate / ridge equivalent)
Other parts unchanged:
- Volatility regime filter
- Half-life filter
- ADF gates ENTRY only
- Robust z-score guards

Adds:
- Output an HTML report (interactive Plotly charts + KPIs + last rows table)
- Optional browser auto-refresh daily via <meta http-equiv="refresh" content="86400">
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt  # kept (unchanged), but Plotly HTML used too

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Bayesian shrinkage regression (conjugate / ridge-form posterior mean)
# =========================
def bayes_shrinkage_alpha_beta(y, x, alpha0=0.0, beta0=1.0,
                               alpha_std=200.0, beta_std=0.10,
                               noise_std=None):
    """
    Model: y = alpha + beta*x + eps, eps ~ N(0, sigma^2)

    Prior: [alpha, beta] ~ N([alpha0, beta0], diag([alpha_std^2, beta_std^2]))
    Posterior mean (MAP == posterior mean for Gaussian-Gaussian):
      m = (X'X/sigma^2 + P0^-1)^-1 (X'y/sigma^2 + P0^-1 m0)

    If noise_std is None, estimate sigma from OLS residuals inside the window.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Design matrix with intercept
    X = np.column_stack([np.ones_like(x), x])  # (n,2)

    # Estimate sigma from OLS if not provided (robust enough for windowed use)
    if noise_std is None:
        ols = sm.OLS(y, X).fit()
        resid = ols.resid
        sigma2 = float(np.var(resid, ddof=1)) if len(resid) > 2 else 1.0
        sigma2 = max(sigma2, 1e-6)
    else:
        sigma2 = float(noise_std) ** 2
        sigma2 = max(sigma2, 1e-6)

    # Prior
    m0 = np.array([alpha0, beta0], dtype=float)
    P0_inv = np.diag([1.0 / (alpha_std ** 2), 1.0 / (beta_std ** 2)])  # (2,2)

    # Posterior precision and mean
    XtX = X.T @ X
    Xty = X.T @ y

    post_prec = (XtX / sigma2) + P0_inv
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ ((Xty / sigma2) + (P0_inv @ m0))

    alpha_hat = float(post_mean[0])
    beta_hat = float(post_mean[1])
    return alpha_hat, beta_hat, float(np.sqrt(sigma2))


# =========================
# 1. Load data
# =========================
file_path = r"C:\Users\bwlau\Desktop\obs.xlsx"
df = pd.read_excel(file_path)

dfA = df.iloc[:, [0, 1]].dropna()
dfA.columns = ["date", "price_A"]
dfB = df.iloc[:, [2, 3]].dropna()
dfB.columns = ["date", "price_B"]

data = pd.merge(dfA, dfB, on="date", how="inner")
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

# Ensure numeric
data["price_A"] = pd.to_numeric(data["price_A"], errors="coerce")
data["price_B"] = pd.to_numeric(data["price_B"], errors="coerce")
data = data.dropna(subset=["price_A", "price_B"]).reset_index(drop=True)


# =========================
# 2. Parameters
# =========================
window = 120
entry_z = 2.5
exit_z = 0.5
adf_threshold = 0.05
contract_size = 25
cost_per_trade = 50   # RM per 1 unit position change (enter/exit). Flip costs 2x.

# Filters
min_half_life = 2
max_half_life = 20
vol_lookback = 252
vol_mult_max = 2.0
min_long_run_std = 1e-6
min_kappa = 1e-6

HOLD_WHEN_FILTER_FAILS = True

# ---- Bayesian prior knobs (THIS is the “Bayesian model”) ----
PRIOR_ALPHA0 = 0.0
PRIOR_BETA0 = 1.0
PRIOR_ALPHA_STD = 500.0   # large => weak shrink on alpha
PRIOR_BETA_STD = 0.10     # smaller => stronger pull toward 1

# ---- HTML report knobs ----
OUTPUT_CSV = "ou_calendar_spread_bayesian_shrinkage.csv"
OUTPUT_HTML = "ou_report.html"
AUTO_REFRESH_DAILY = True  # adds meta refresh (86400 seconds)
REFRESH_SECONDS = 86400


# =========================
# 3. Containers
# =========================
for c in ["alpha", "beta", "theta", "kappa", "sigma", "spread", "z", "adf_p",
          "half_life", "spread_vol", "baseline_vol", "trade_ok"]:
    data[c] = np.nan

data["position"] = 0
data["pnl"] = 0.0
data["equity"] = 0.0


# =========================
# Helper: robust baseline vol
# =========================
def robust_baseline_vol(series: pd.Series, win: int = 120) -> float:
    roll = series.rolling(win, min_periods=max(20, win // 3)).std()
    roll = roll.replace([np.inf, -np.inf], np.nan).dropna()
    if len(roll) == 0:
        return np.nan
    return float(np.nanmedian(roll.values))


# =========================
# 4. Rolling Estimation
# =========================
for i in range(window, len(data)):
    hist = data.iloc[i - window:i].dropna(subset=["price_A", "price_B"])
    if len(hist) < 60:
        continue

    y = hist["price_A"].values
    x = hist["price_B"].values

    # 1) Bayesian shrinkage hedge: A = alpha + beta * B
    alpha, beta, _ = bayes_shrinkage_alpha_beta(
        y, x,
        alpha0=PRIOR_ALPHA0, beta0=PRIOR_BETA0,
        alpha_std=PRIOR_ALPHA_STD, beta_std=PRIOR_BETA_STD,
        noise_std=None
    )

    # 2) Spread in price space
    spread_hist = hist["price_A"] - (alpha + beta * hist["price_B"])
    spread_hist = spread_hist.replace([np.inf, -np.inf], np.nan).dropna()
    if len(spread_hist) < 60:
        continue

    # 3) Vol regime filter
    spread_vol = float(np.std(spread_hist.values, ddof=1))

    start_idx = max(0, i - vol_lookback)
    hist_long = data.iloc[start_idx:i].dropna(subset=["price_A", "price_B"])
    baseline_vol = np.nan
    if len(hist_long) >= 60:
        baseline_spread = hist_long["price_A"] - (alpha + beta * hist_long["price_B"])
        baseline_vol = robust_baseline_vol(baseline_spread, win=min(120, len(baseline_spread)))

    data.loc[i, "spread_vol"] = spread_vol
    data.loc[i, "baseline_vol"] = baseline_vol

    if (not np.isnan(baseline_vol)) and (spread_vol > vol_mult_max * baseline_vol):
        data.loc[i, "trade_ok"] = 0
        data.loc[i, ["alpha", "beta"]] = alpha, beta
        continue

    # 4) ADF test (record p-value)
    try:
        adf_p = float(adfuller(spread_hist.values, autolag="AIC")[1])
    except Exception:
        continue
    data.loc[i, "adf_p"] = adf_p

    # 5) OU fit: X_{t+1} = a + b X_t + e
    X_t = spread_hist.values[:-1]
    X_t1 = spread_hist.values[1:]
    ou_reg = sm.OLS(X_t1, sm.add_constant(X_t, has_constant="add")).fit()
    a = float(ou_reg.params[0])
    b = float(ou_reg.params[1])

    if not (0 < b < 0.99):
        continue

    kappa = float(-np.log(b))
    if (not np.isfinite(kappa)) or (kappa < min_kappa):
        continue

    theta = float(a / (1 - b))
    resid = ou_reg.resid
    resid_std = float(np.std(resid, ddof=1))
    sigma = resid_std * np.sqrt(2 * kappa / (1 - b**2))

    half_life = float(np.log(2) / kappa) if kappa > 0 else np.nan
    data.loc[i, "half_life"] = half_life

    if np.isnan(half_life) or (half_life < min_half_life) or (half_life > max_half_life):
        data.loc[i, "trade_ok"] = 0
        data.loc[i, ["alpha", "beta", "theta", "kappa", "sigma"]] = alpha, beta, theta, kappa, sigma
        continue

    # 6) current spread & z-score
    spread_now = float(data.loc[i, "price_A"] - (alpha + beta * data.loc[i, "price_B"]))
    long_run_std = sigma / np.sqrt(2 * kappa)

    if (not np.isfinite(long_run_std)) or (long_run_std < min_long_run_std):
        data.loc[i, "trade_ok"] = 0
        data.loc[i, ["alpha", "beta", "theta", "kappa", "sigma", "spread"]] = alpha, beta, theta, kappa, sigma, spread_now
        continue

    z = float((spread_now - theta) / long_run_std)
    z = float(np.clip(z, -10, 10))

    data.loc[i, "trade_ok"] = 1
    data.loc[i, ["alpha", "beta", "theta", "kappa", "sigma", "spread", "z"]] = \
        alpha, beta, theta, kappa, sigma, spread_now, z


# =========================
# 5. Trading Logic (Signal at t, Execute at t+1)
# =========================
pos = 0
for i in range(1, len(data)):
    z = data.loc[i, "z"]
    adf_p = data.loc[i, "adf_p"]
    trade_ok = data.loc[i, "trade_ok"]

    model_ok_today = (not np.isnan(z)) and (trade_ok == 1)

    if not np.isnan(z) and abs(z) < exit_z:
        pos = 0
    else:
        if model_ok_today:
            entry_allowed = (not np.isnan(adf_p)) and (adf_p < adf_threshold)
            if entry_allowed:
                if z > entry_z:
                    pos = -1
                elif z < -entry_z:
                    pos = 1
            else:
                if not HOLD_WHEN_FILTER_FAILS:
                    pos = 0
        else:
            if not HOLD_WHEN_FILTER_FAILS:
                pos = 0

    data.loc[i, "position"] = pos


# =========================
# 6. PnL (T+1 execution, RM)
# =========================
dA = data["price_A"].diff()
dB = data["price_B"].diff()

beta_ff = data["beta"].shift(1).ffill().fillna(0.0)
pos_exec = data["position"].shift(1).fillna(0)

raw_pnl = pos_exec * (dA - beta_ff * dB) * contract_size

trades = data["position"].diff().abs().fillna(0)
cost = trades * cost_per_trade

data["pnl"] = raw_pnl.fillna(0) - cost
data["equity"] = data["pnl"].cumsum()


# =========================
# 7. Results
# =========================
print("Total PnL (RM):", float(data["equity"].iloc[-1]))
print("Number of trades:", int(trades.sum()))
print("Beta median/std:", float(data["beta"].median()), float(data["beta"].std()))

data.to_csv(OUTPUT_CSV, index=False)


# =========================
# 8. Matplotlib Plots (unchanged)
# =========================
plt.figure(figsize=(14, 12))

plt.subplot(6, 1, 1)
plt.plot(data["date"], data["spread"], label="Spread")
plt.plot(data["date"], data["theta"], "--", label="OU Mean")
plt.legend(); plt.title("Price Spread (Bayesian Shrinkage Hedge)")

plt.subplot(6, 1, 2)
plt.plot(data["date"], data["beta"], label="beta (shrunken)")
plt.axhline(PRIOR_BETA0, color="k", ls=":", label="prior beta0")
plt.legend(); plt.title("Beta (Bayesian shrinkage toward 1.0)")

plt.subplot(6, 1, 3)
plt.plot(data["date"], data["z"], label="Z-score")
plt.axhline(entry_z, color="r", ls="--")
plt.axhline(-entry_z, color="r", ls="--")
plt.axhline(exit_z, color="g", ls="--")
plt.axhline(-exit_z, color="g", ls="--")
plt.axhline(0, color="k", ls=":")
plt.legend(); plt.title("Z-score")

plt.subplot(6, 1, 4)
plt.plot(data["date"], data["adf_p"], label="ADF p-value")
plt.axhline(adf_threshold, color="r", ls="--", label="ADF Threshold")
plt.legend(); plt.title("ADF Stationarity Filter (Entry gate)")

plt.subplot(6, 1, 5)
plt.plot(data["date"], data["half_life"], label="Half-life (days)")
plt.axhline(max_half_life, color="r", ls="--", label="Max half-life")
plt.legend(); plt.title("Half-life")

plt.subplot(6, 1, 6)
plt.plot(data["date"], data["equity"], label="Equity (RM)", color="darkgreen")
plt.legend(); plt.title("Equity Curve (RM)")

plt.tight_layout()
plt.show()


# =========================
# 9. HTML Report (ADDED)
# =========================
# Latest metrics
last = data.iloc[-1]
last_date = str(last["date"])
last_z = float(last["z"]) if pd.notna(last["z"]) else np.nan
last_pos = int(last["position"]) if pd.notna(last["position"]) else 0
last_beta = float(last["beta"]) if pd.notna(last["beta"]) else np.nan
last_eq = float(last["equity"]) if pd.notna(last["equity"]) else 0.0
last_adf = float(last["adf_p"]) if pd.notna(last["adf_p"]) else np.nan
last_hl = float(last["half_life"]) if pd.notna(last["half_life"]) else np.nan

num_trades = int(data["position"].diff().abs().fillna(0).sum())

kpis_html = f"""
<h2>OU Calendar Spread Report</h2>
<ul>
  <li><b>Last date:</b> {last_date}</li>
  <li><b>Last z:</b> {last_z:.2f}</li>
  <li><b>Last position:</b> {last_pos}</li>
  <li><b>Last beta:</b> {last_beta:.4f}</li>
  <li><b>Last ADF p:</b> {last_adf:.4f}</li>
  <li><b>Last half-life:</b> {last_hl:.2f}</li>
  <li><b>Total equity (RM):</b> {last_eq:.2f}</li>
  <li><b>Trades (count):</b> {num_trades}</li>
</ul>
<hr>
"""

# Build interactive charts
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("Equity (RM)", "Z-score", "Spread vs OU Mean")
)

fig.add_trace(go.Scatter(x=data["date"], y=data["equity"], name="Equity"), row=1, col=1)

fig.add_trace(go.Scatter(x=data["date"], y=data["z"], name="Z"), row=2, col=1)
fig.add_hline(y=entry_z, line_dash="dash", row=2, col=1)
fig.add_hline(y=-entry_z, line_dash="dash", row=2, col=1)
fig.add_hline(y=exit_z, line_dash="dash", row=2, col=1)
fig.add_hline(y=-exit_z, line_dash="dash", row=2, col=1)
fig.add_hline(y=0, line_dash="dot", row=2, col=1)

fig.add_trace(go.Scatter(x=data["date"], y=data["spread"], name="Spread"), row=3, col=1)
fig.add_trace(go.Scatter(x=data["date"], y=data["theta"], name="OU Mean", line_dash="dash"), row=3, col=1)

fig.update_layout(height=900, title_text="OU Spread Dashboard (EOD)")

# Recent rows table
tail = data[["date","price_A","price_B","beta","spread","z","adf_p","half_life",
             "position","pnl","equity"]].tail(50)

# Format floats safely
def _fmt(x):
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4f}"
        return str(x)
    except Exception:
        return str(x)

tail_fmt = tail.copy()
for col in tail_fmt.columns:
    tail_fmt[col] = tail_fmt[col].map(_fmt)

table_html = tail_fmt.to_html(index=False, escape=False)

meta_refresh = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">' if AUTO_REFRESH_DAILY else ""

html = f"""
<html>
<head>
  <meta charset="utf-8">
  {meta_refresh}
  <title>OU Spread Report</title>
</head>
<body>
  {kpis_html}
  {fig.to_html(full_html=False, include_plotlyjs="cdn")}
  <h3>Last 50 rows</h3>
  {table_html}
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved CSV: {OUTPUT_CSV}")
print(f"Saved HTML report: {OUTPUT_HTML}")
