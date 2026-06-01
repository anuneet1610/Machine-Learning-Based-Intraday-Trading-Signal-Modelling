import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from clickhouse_driver import Client
from pathlib import Path
import logging

# ============================================================================
# Configuration — edit these
# ============================================================================
MODEL_PATH          = "xgb_big_move_model.json"
SYMBOL_MAPPING_PATH = "symbol_mapping.pkl"
LOG_PATH            = "../logs/backtest.log"

THRESHOLD           = 0.35      # probability threshold to trigger a trade signal
TRANSACTION_COST    = 0.0005    # 5bps per side
SLIPPAGE            = 0.0002    # 2bps slippage per side
SENTIMENT_WINDOW    = 89        # must match training

TEST_START          = "2024-01-02"
TEST_END            = "2024-12-30"

# ============================================================================
# Logging
# ============================================================================
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("backtest")
logger.info("Starting backtest")

# ============================================================================
# Load model
# ============================================================================
print("Loading model...")
mag_model = xgb.XGBClassifier(); mag_model.load_model("xgb_big_move_model.json")
dir_model = xgb.XGBClassifier(); dir_model.load_model("xgb_dir_model.json")

# ============================================================================
# Load symbol mapping
# ============================================================================
with open(SYMBOL_MAPPING_PATH, "rb") as f:
    symbol_mapping = pickle.load(f)  # int -> symbol string
reverse_mapping = {v: k for k, v in symbol_mapping.items()}
logger.info(f"Loaded symbol mapping: {len(symbol_mapping)} symbols")

# ============================================================================
# ClickHouse — fetch test period data
# ============================================================================
print("Connecting to ClickHouse...")
try:
    client = Client(host='localhost')
    logger.info("Connected to ClickHouse")
except Exception as e:
    logger.error(f"ClickHouse connection failed: {e}", exc_info=True)
    raise

query = f"""
    SELECT
        symbol, datetime, close, sigma_forecast, arma_forecast,
        ema_trend_filter_trend_up, ema_trend_filter_trend_down,
        long_term_bias_trend_up, long_term_bias_trend_down,
        macd_signal, risk_adj_ret, long_signal, short_signal,
        rsi_timing, pct_change, weighted_avg_sentiment_api,
        weighted_avg_sentiment_llm, hour, day_of_week, day_of_month,
        month, quarter, year, hour_sin, hour_cos, day_sin, day_cos,
        month_sin, month_cos
    FROM final_table
    WHERE datetime >= toDateTime('2024-01-02 00:00:00')
    AND datetime <= toDateTime('2024-12-30 23:59:59')
    ORDER BY datetime ASC, symbol
"""

print("Fetching test data from ClickHouse...")
try:
    client.execute("USE market_data")
    raw = client.execute(query)
    logger.info(f"Fetched {len(raw)} rows")
except Exception as e:
    logger.error(f"Query failed: {e}", exc_info=True)
    raise

columns = [
    'symbol', 'datetime', 'close', 'sigma_forecast', 'arma_forecast',
    'ema_trend_filter_trend_up', 'ema_trend_filter_trend_down',
    'long_term_bias_trend_up', 'long_term_bias_trend_down',
    'macd_signal', 'risk_adj_ret', 'long_signal', 'short_signal',
    'rsi_timing', 'pct_change', 'weighted_avg_sentiment_api',
    'weighted_avg_sentiment_llm', 'hour', 'day_of_week', 'day_of_month',
    'month', 'quarter', 'year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month_sin', 'month_cos'
]

df = pd.DataFrame(raw, columns=columns)
df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
print(f"Loaded {len(df):,} rows | {df['datetime'].min()} → {df['datetime'].max()}")

# ============================================================================
# Feature engineering — replicate training transforms
# ============================================================================
print("Engineering features...")

# Sentiment z-score (same window as training)
rolling_mean = (
    df.groupby('symbol')['weighted_avg_sentiment_api']
      .transform(lambda x: x.shift(1).rolling(SENTIMENT_WINDOW, min_periods=20).mean())
)
rolling_std = (
    df.groupby('symbol')['weighted_avg_sentiment_api']
      .transform(lambda x: x.shift(1).rolling(SENTIMENT_WINDOW, min_periods=20).std())
)
df['sentiment_zscore'] = (
    (df['weighted_avg_sentiment_api'] - rolling_mean) / (rolling_std + 1e-8)
)

# Map symbol strings to integer codes
df['symbol'] = df['symbol'].map(reverse_mapping)
unmapped = df['symbol'].isna().sum()
if unmapped > 0:
    logger.warning(f"{unmapped} rows have unmapped symbols — dropping")
    df = df.dropna(subset=['symbol'])
df['symbol'] = df['symbol'].astype(int)

# Ground-truth binary target (for reference only — not used in signal generation)
# df['up_down'] = (df['pct_change'].abs() > 0.2).astype(int)

df['target'] = np.where(df['pct_change'] >  0.2, 2,
               np.where(df['pct_change'] < -0.2, 0, 1))

# ============================================================================
# Prepare feature matrix
# ============================================================================
# drop_cols = ['pct_change', 'up_down', 'datetime']
drop_cols = ['pct_change', 'target', 'datetime']
feature_cols = [c for c in df.columns if c not in drop_cols]

# Run on df BEFORE train/test split, one symbol only
one_sym = df[df['symbol'] == df['symbol'].iloc[0]].copy().reset_index(drop=True)

from sklearn.metrics import roc_auc_score
import numpy as np

print(f"{'N':>5} {'AUC (macd)':>12} {'AUC (ema_up)':>13} {'AUC (rsi)':>10} {'class_balance':>14}")
print("-" * 60)

for N in [1, 3, 6, 12, 24, 39, 78]:
    fwd = sum(one_sym['pct_change'].shift(-i) for i in range(1, N + 1))
    label = (fwd > 0).astype(int)
    mask = fwd.notna()

    try:
        auc_macd = roc_auc_score(label[mask], one_sym.loc[mask, 'macd_signal'])
        auc_ema = roc_auc_score(label[mask], one_sym.loc[mask, 'ema_trend_filter_trend_up'])
        auc_rsi = roc_auc_score(label[mask], one_sym.loc[mask, 'rsi_timing'])
        balance = label[mask].mean()
        print(f"{N:>5} {auc_macd:>12.4f} {auc_ema:>13.4f} {auc_rsi:>10.4f} {balance:>14.4f}")
    except Exception as e:
        print(f"{N:>5} error: {e}")

X = df[feature_cols].copy()
# meta = df[['datetime', 'symbol', 'pct_change', 'long_signal', 'short_signal', 'up_down']].copy()
meta = df[['datetime', 'symbol', 'pct_change', 'long_signal', 'short_signal', 'target']].copy()

# Drop rows with NaN or Inf in features
numeric_cols = X.select_dtypes(include=[np.number]).columns
clean_mask = ~(
    X.isna().any(axis=1) |
    np.isinf(X[numeric_cols]).any(axis=1) |
    meta['pct_change'].isna() |
    np.isinf(meta['pct_change'])
)
X    = X[clean_mask].reset_index(drop=True)
meta = meta[clean_mask].reset_index(drop=True)
print(f"Clean rows for inference: {len(X):,}")

# # ============================================================================
# # Inference
# # ============================================================================
# print("Running model inference...")
# meta['prob']   = model.predict_proba(X)[:, 1]
# meta['signal'] = (meta['prob'] > THRESHOLD).astype(int)
#
# # ============================================================================
# # Direction — combine model signal with existing directional signals
# # ============================================================================
# # Big move predicted AND long_signal=1  → long (+1)
# # Big move predicted AND short_signal=1 → short (-1)
# # No signal or no direction             → flat (0)
# meta['direction'] = np.where(
#     meta['long_signal'] == 1,  1,
#     np.where(meta['short_signal'] == 1, -1, 0)
# )
# meta['trade_signal'] = meta['signal'] * meta['direction']

# ============================================================================
# Compute 12-bar forward return (must match training target)
# ============================================================================
meta = meta.sort_values(['symbol', 'datetime']).reset_index(drop=True)

meta['forward_ret_12'] = (
    meta.groupby('symbol')['pct_change']
        .transform(
            lambda x: sum(x.shift(-i) for i in range(1, 13))
        )
) / 100.0

# Drop rows where we can't compute full 12-bar forward window
# Use .values to get a numpy bool array — index-agnostic, avoids pandas alignment error
valid_idx = meta['forward_ret_12'].notna().values
meta = meta[valid_idx].reset_index(drop=True)
X    = X[valid_idx].reset_index(drop=True)

# ============================================================================
# Signal generation (same as before)
# ============================================================================
meta['prob_big'] = mag_model.predict_proba(X)[:, 1]
meta['prob_up']  = dir_model.predict_proba(X)[:, 1]

meta['signal']    = (meta['prob_big'] > THRESHOLD).astype(int)
meta['direction'] = np.where(meta['prob_up'] > 0.53,  1,
                    np.where(meta['prob_up'] < 0.42, -1, 0))
meta['trade_signal'] = meta['signal'] * meta['direction']

print(meta['prob_up'].describe())
print(f"prob_up > 0.55 : {(meta['prob_up'] > 0.55).sum():,}")
print(f"prob_up < 0.45 : {(meta['prob_up'] < 0.45).sum():,}")
print(f"trade_signal != 0 : {(meta['trade_signal'] != 0).sum():,}")

# ============================================================================
# Non-overlapping trade execution per symbol
# ============================================================================
HOLD_BARS = 12

def apply_holding_period(sym_df, hold=HOLD_BARS):
    sym_df = sym_df.copy().reset_index(drop=True)
    pnl    = np.zeros(len(sym_df))
    i = 0
    while i < len(sym_df):
        if sym_df.loc[i, 'trade_signal'] != 0:
            direction = sym_df.loc[i, 'trade_signal']
            fwd_ret   = sym_df.loc[i, 'forward_ret_12']
            gross     = direction * fwd_ret
            cost      = TRANSACTION_COST + SLIPPAGE  # one entry + one exit
            pnl[i]    = gross - cost
            i += hold  # skip forward, no overlapping trades
        else:
            i += 1
    sym_df['net_pnl'] = pnl
    return sym_df

# Explicit loop avoids pandas 2.x groupby dropping the key column in apply()
result_parts = []
for sym_code in meta['symbol'].unique():
    sym_df = meta[meta['symbol'] == sym_code].copy()
    result_parts.append(apply_holding_period(sym_df))
meta = pd.concat(result_parts).sort_values(['symbol', 'datetime']).reset_index(drop=True)

# ============================================================================
# Stats (unchanged — your existing backtest_stats function works as-is)
# ============================================================================
# Update BARS_PER_YEAR since effective "trades" are now spaced ~12 bars apart
EFFECTIVE_BARS = 252 * (78 // HOLD_BARS)  # ~252 * 6 = 1512 trade slots per year

def backtest_stats(pnl, label=""):
    traded    = pnl[pnl != 0]
    cum       = (1 + pnl).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_trades  = len(traded)
    hit_rate  = (traded > 0).mean() if n_trades > 0 else np.nan
    sharpe    = (pnl.mean() / (pnl.std() + 1e-10)) * np.sqrt(EFFECTIVE_BARS)
    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = drawdown.min()
    gp        = traded[traded > 0].sum()
    gl        = traded[traded < 0].abs().sum()
    pf        = gp / (gl + 1e-10)

    print(f"\n{'='*45}")
    print(f"  {label}")
    print(f"{'='*45}")
    print(f"  Total Return  : {total_ret*100:+.2f}%")
    print(f"  Sharpe Ratio  : {sharpe:.3f}")
    print(f"  Max Drawdown  : {max_dd*100:.2f}%")
    print(f"  # Trades      : {n_trades:,}")
    print(f"  Hit Rate      : {hit_rate*100:.1f}%")
    print(f"  Profit Factor : {pf:.3f}")

    return dict(label=label, total_return=total_ret, sharpe=sharpe,
                max_drawdown=max_dd, n_trades=n_trades,
                hit_rate=hit_rate, profit_factor=pf)

# ============================================================================
# Aggregate + per-symbol stats
# ============================================================================
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"  Threshold    : {THRESHOLD}")
print(f"  Trans. cost  : {TRANSACTION_COST*10000:.1f} bps/side")
print(f"  Slippage     : {SLIPPAGE*10000:.1f} bps/side")
print(f"  Period       : {TEST_START} → {TEST_END}")

all_stats  = backtest_stats(meta['net_pnl'], "PORTFOLIO (all symbols)")
sym_stats  = {}
sym_curves = {}

for sym_code in sorted(meta['symbol'].unique()):
    sym_name = symbol_mapping.get(sym_code, str(sym_code))
    sym_df   = meta[meta['symbol'] == sym_code]
    stats    = backtest_stats(sym_df['net_pnl'], sym_name)
    sym_stats[sym_name]  = stats
    sym_curves[sym_name] = (1 + sym_df['net_pnl']).cumprod().values

# ============================================================================
# Threshold sensitivity table
# ============================================================================
print("\n" + "="*60)
print("THRESHOLD SENSITIVITY")
print("="*60)
print(f"{'Threshold':>10} {'Trades':>8} {'Hit%':>7} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8}")
print("-" * 55)

for t in np.arange(0.10, 0.71, 0.05):
    sig   = (meta['prob_big'] > t).astype(int) * meta['direction']
    gross = sig * meta['pct_change'] / 100.0
    cost  = sig.abs() * (TRANSACTION_COST + SLIPPAGE)
    pnl   = gross - cost
    traded = pnl[pnl != 0]
    n      = len(traded)
    if n == 0:
        continue
    hr  = (traded > 0).mean() * 100
    sh  = (pnl.mean() / (pnl.std() + 1e-10)) * np.sqrt(EFFECTIVE_BARS)
    ret = ((1 + pnl).cumprod().iloc[-1] - 1) * 100
    mdd = ((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min() * 100
    marker = " ◄" if abs(t - THRESHOLD) < 0.001 else ""
    print(f"{t:>10.2f} {n:>8,} {hr:>7.1f} {sh:>8.3f} {ret:>9.2f} {mdd:>8.2f}{marker}")

# ============================================================================
# Plots
# ============================================================================
print("\nGenerating plots...")

portfolio_cum = (1 + meta['net_pnl']).cumprod()
roll_max      = portfolio_cum.cummax()
drawdown_ser  = (portfolio_cum - roll_max) / roll_max
rolling_sharpe = (
    meta['net_pnl']
      .rolling(EFFECTIVE_BARS // 12)   # ~1 month
      .apply(lambda x: (x.mean() / (x.std() + 1e-10)) * np.sqrt(EFFECTIVE_BARS), raw=True)
)

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1. Equity curve — all symbols + portfolio
ax1 = fig.add_subplot(gs[0, :])
for sym_name, curve in sym_curves.items():
    ax1.plot(curve, lw=1.2, alpha=0.75, label=sym_name)
ax1.plot(portfolio_cum.values, color='black', lw=2.5, linestyle='--', label='Portfolio')
ax1.set_title(f'Equity Curve — Net of Costs (threshold={THRESHOLD})')
ax1.set_ylabel('Cumulative Return (1 = start)')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(range(len(drawdown_ser)), drawdown_ser.values, 0,
                 color='crimson', alpha=0.45)
ax2.set_title('Portfolio Drawdown')
ax2.set_ylabel('Drawdown')
ax2.grid(True, alpha=0.3)

# 3. Rolling Sharpe
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(rolling_sharpe.values, color='steelblue', lw=1.5)
ax3.axhline(0, color='black', lw=0.8)
ax3.axhline(1, color='green',  lw=0.8, linestyle='--', alpha=0.6, label='Sharpe = 1')
ax3.axhline(-1, color='red',   lw=0.8, linestyle='--', alpha=0.6, label='Sharpe = -1')
ax3.set_title('Rolling 1-Month Sharpe Ratio')
ax3.set_ylabel('Sharpe')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. PnL distribution
ax4 = fig.add_subplot(gs[2, 0])
traded_pnl = meta.loc[meta['net_pnl'] != 0, 'net_pnl'] * 100
ax4.hist(traded_pnl, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
ax4.axvline(0, color='black', lw=1.2)
ax4.axvline(traded_pnl.mean(), color='orange', lw=1.5, linestyle='--',
            label=f'Mean={traded_pnl.mean():.3f}%')
ax4.set_title('Trade PnL Distribution')
ax4.set_xlabel('Net PnL per bar (%)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Per-symbol summary bar chart
ax5 = fig.add_subplot(gs[2, 1])
sym_returns = {k: v['total_return'] * 100 for k, v in sym_stats.items()}
colors = ['green' if v >= 0 else 'crimson' for v in sym_returns.values()]
ax5.bar(sym_returns.keys(), sym_returns.values(), color=colors, edgecolor='white', alpha=0.85)
ax5.axhline(0, color='black', lw=0.8)
ax5.set_title('Total Return by Symbol (%)')
ax5.set_ylabel('Return (%)')
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'XGBoost Big Move Backtest  |  {TEST_START} → {TEST_END}  |  '
    f'Threshold={THRESHOLD}  |  Costs={int((TRANSACTION_COST+SLIPPAGE)*10000)}bps/side',
    fontsize=11, y=1.01
)

out_path = "backtest_results.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to {out_path}")
plt.show()

logger.info("Backtest complete")
print("\n✓ Done.")