"""
Calculation Service - Stock Market Technical Analysis Microservice

This service performs real-time technical analysis on stock market data:
- GARCH volatility forecasting
- ARMA return forecasting
- RSI (Relative Strength Index) calculations
- EMA (Exponential Moving Average) calculations
- MACD (Moving Average Convergence Divergence) signal generation
- Trading signal generation based on multiple indicators

Input: Kafka topic 'stock_table' with stock quote data
Output: Kafka topic 'stock_calculation_table' with enriched technical indicators
"""
import pathway as pw
import math
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Service configuration
MICROSERVICE_NAME = "calc_service"

# Ensure logs directory exists
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging for production
# Using 'a' (append) mode instead of 'w' to preserve logs across restarts
logging.basicConfig(
    level=logging.INFO,
    filename=f"../logs/{MICROSERVICE_NAME}.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(MICROSERVICE_NAME)
logger.info(f"Starting {MICROSERVICE_NAME} - Initializing technical analysis pipeline")


class QuoteSchema(pw.Schema):
    """
    Schema for stock quote data from Kafka.
    
    Attributes:
        timestamp: Human-readable timestamp string
        symbol: Stock ticker symbol (e.g., 'AAPL')
        open: Opening price for the period
        high: Highest price during the period
        low: Lowest price during the period
        close: Closing price for the period
        volume: Trading volume (number of shares)
        ts_ms: Timestamp in milliseconds (used for temporal operations)
    """
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ts_ms: int


# ============================================================================
# KAFKA INPUT: Read stock quote data from Kafka topic
# ============================================================================
kafka_broker = os.getenv("KAFKA_BROKER")
if not kafka_broker:
    logger.error("KAFKA_BROKER environment variable not set - cannot connect to Kafka")
    raise ValueError("KAFKA_BROKER environment variable is required")

logger.info(f"Connecting to Kafka broker: {kafka_broker}")
logger.info("Reading from Kafka topic 'stock_table'...")

try:
    quotes = pw.io.kafka.read(
        rdkafka_settings={
            "bootstrap.servers": kafka_broker,
            "group.id": "pathway-group",
            "auto.offset.reset": "earliest",  # Start from beginning if no offset exists
        },
        topic="stock_table",
        format="json",
        schema=QuoteSchema,
        json_field_paths={
            "symbol": "/value/symbol",
            "open": "/value/open",
            "high": "/value/high",
            "low": "/value/low",
            "close": "/value/close",
            "volume": "/value/volume",
            "timestamp": "/value/timestamp",
            "ts_ms": "/value/ts_ms"
        }
    )
    logger.info("Successfully configured Kafka reader for topic 'stock_table'")
except Exception as e:
    logger.error(f"Failed to configure Kafka reader: {str(e)}", exc_info=True)
    raise

# Prepare price tuples for temporal windowing operations
# Tuple format: (timestamp_ms, close_price) for efficient sorting and processing
logger.debug("Preparing price tuples for temporal operations")
quotes = quotes.with_columns(
    ts_price_tuple=pw.make_tuple(quotes.ts_ms, quotes.close)
)

# ============================================================================
# OPTIMIZATION 1: GARCH Volatility Forecasting + ARMA Return Forecasting
# ============================================================================
# Combine prev_close, returns, and GARCH calculations in a single window
# to reduce computational overhead and improve performance.
#
# Window Configuration:
# - Hop: 5 minutes (5 * 60,000 ms) - window slides every 5 minutes
# - Duration: 150 minutes (30 periods * 5 min) - window contains 30 data points
# - Cutoff: 175 minutes - maximum delay before window is considered complete
#
# GARCH Model Parameters:
# - omega: Long-term variance (1e-6)
# - alpha: Weight for recent squared residuals (0.05)
# - beta: Weight for previous variance (0.93)
# - phi: AR coefficient for ARMA model (0.6)
# - theta: MA coefficient for ARMA model (0.3)
logger.info("Creating sliding window for GARCH volatility and ARMA return forecasting")
logger.debug("Window: 30 periods (150 min), hop: 5 min, cutoff: 175 min")

windowed_combined = quotes.windowby(
    quotes.ts_ms,
    window=pw.temporal.sliding(
        hop=5 * 60_000,  # 5 minutes
        duration=30 * 5 * 60_000,  # 30 periods = 150 minutes
    ),
    instance=quotes.symbol,  # Group by stock symbol
    behavior=pw.temporal.common_behavior(cutoff=35 * 5 * 60_000),  # 175 minutes max delay
).reduce(
    symbol=pw.this._pw_instance,
    price_close_tuples=pw.reducers.sorted_tuple(pw.make_tuple(pw.this.ts_ms, pw.this.close)),
    cnt=pw.reducers.count(),  # Track number of data points in window
    start_ts=pw.this._pw_window_start,
    end_ts=pw.this._pw_window_end,
)

logger.info("GARCH window created - waiting for sufficient data points...")
# ============================================================================
# GARCH + ARMA Calculation UDF
# ============================================================================
@pw.udf
def calculate_all_metrics(price_tuples: tuple,
                          omega: float = 1e-6,
                          alpha: float = 0.05,
                          beta: float = 0.93,
                          phi: float = 0.6,
                          theta: float = 0.3,
                          init_sigma: float = 1e-3) -> tuple[float, float, float, float, float, float]:
    """
    Calculate previous close, returns, GARCH volatility forecast, and ARMA return forecast.
    
    This function combines multiple calculations to optimize performance:
    - Previous period's closing price
    - Current period's log return
    - GARCH(1,1) volatility forecast (sigma_forecast)
    - ARMA(1,1) return forecast
    - Current volatility estimate (sigma_t)
    - Current residual (actual return - ARMA forecast)
    
    Args:
        price_tuples: List of (timestamp_ms, price) tuples, sorted by timestamp
        omega: GARCH long-term variance parameter (default: 1e-6)
        alpha: GARCH weight for recent squared residuals (default: 0.05)
        beta: GARCH weight for previous variance (default: 0.93)
        phi: ARMA autoregressive coefficient (default: 0.6)
        theta: ARMA moving average coefficient (default: 0.3)
        init_sigma: Initial volatility estimate (default: 1e-3)
    
    Returns:
        Tuple of (prev_close, current_ret, sigma_forecast, arma_forecast, sigma_t, resid):
        - prev_close: Previous period's closing price
        - current_ret: Current period's log return
        - sigma_forecast: Forecasted volatility for next period
        - arma_forecast: Forecasted return for next period
        - sigma_t: Current period's volatility estimate
        - resid: Current period's residual (actual - forecasted return)
    
    Note:
        Returns default values if insufficient data is available.
        GARCH model: sigma^2(t+1) = omega + alpha * resid^2(t) + beta * sigma^2(t)
        ARMA model: ret(t+1) = phi * ret(t) + theta * resid(t)
    """
    if not price_tuples or len(price_tuples) == 0:
        logger.warning("calculate_all_metrics: Empty price_tuples, returning default values")
        return (0.0, 0.0, init_sigma, 0.0, init_sigma, 0.0)

    # Extract prices from tuples (ignore timestamps for calculation)
    prices = [p for (_, p) in price_tuples]

    # Get previous period's closing price (second-to-last price)
    # If only one price available, use it as prev_close
    prev_close = prices[-2] if len(prices) >= 2 else prices[-1]

    # Calculate current period's log return: log(price_t / price_{t-1})
    # Log returns are preferred for financial modeling due to time-additivity
    if len(prices) >= 2 and prices[-2] > 0 and prices[-1] > 0:
        current_ret = math.log(prices[-1] / prices[-2])
    else:
        current_ret = 0.0
        if len(prices) >= 2:
            logger.warning(f"calculate_all_metrics: Invalid prices for return calculation: {prices[-2]}, {prices[-1]}")

    # Need at least 2 prices to calculate returns
    if len(prices) < 2:
        logger.warning("calculate_all_metrics: Insufficient prices for GARCH calculation")
        return (prev_close, current_ret, init_sigma, 0.0, init_sigma, 0.0)

    # Calculate log returns for all price pairs in the window
    rets = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            rets.append(math.log(prices[i] / prices[i - 1]))
        else:
            rets.append(0.0)
            logger.warning(f"calculate_all_metrics: Invalid price pair at index {i}: {prices[i-1]}, {prices[i]}")

    # Initialize GARCH model with sample variance
    # Use mean return and variance of historical returns as starting point
    mean_ret = sum(rets) / len(rets) if rets else 0.0
    var = sum((x - mean_ret) ** 2 for x in rets) / len(rets) if rets else init_sigma * init_sigma
    last_sigma2 = var if var > 0 else init_sigma * init_sigma

    # Initialize ARMA model state
    prev_ret = rets[0] if rets else 0.0
    last_resid = 0.0

    # Iterate through returns to update GARCH and ARMA models
    # This implements the recursive GARCH(1,1) and ARMA(1,1) updates
    for i in range(1, len(rets)):
        cur_ret = rets[i]
        
        # ARMA forecast: ret(t) = phi * ret(t-1) + theta * resid(t-1)
        pred = phi * prev_ret + theta * last_resid
        
        # Residual: actual return - ARMA forecast
        resid = cur_ret - pred
        
        # GARCH variance update: sigma^2(t) = omega + alpha * resid^2(t-1) + beta * sigma^2(t-1)
        sigma2 = omega + alpha * (last_resid ** 2) + beta * last_sigma2
        
        # Update state for next iteration
        last_sigma2 = sigma2
        last_resid = resid
        prev_ret = cur_ret

    # Forecast next period's volatility and return
    sigma_forecast2 = omega + alpha * (last_resid ** 2) + beta * last_sigma2
    arma_forecast = phi * prev_ret + theta * last_resid

    # Return all metrics (convert variance to standard deviation)
    return (prev_close, current_ret, math.sqrt(sigma_forecast2), arma_forecast, math.sqrt(last_sigma2), last_resid)


# Apply GARCH and ARMA calculations to windowed data
logger.info("Applying GARCH volatility and ARMA return forecasting calculations")
windowed_combined = windowed_combined.with_columns(
    metrics=calculate_all_metrics(windowed_combined.price_close_tuples)
).with_columns(
    # Extract individual metrics from the tuple result
    prev_close=pw.this.metrics[0],
    ret=pw.this.metrics[1],
    sigma_forecast=pw.this.metrics[2],
    arma_forecast=pw.this.metrics[3],
    sigma_t=pw.this.metrics[4],
    resid=pw.this.metrics[5],
).select(
    symbol=pw.this.symbol,
    prev_close=pw.this.prev_close,
    ret=pw.this.ret,
    sigma_forecast=pw.this.sigma_forecast,
    arma_forecast=pw.this.arma_forecast,
    sigma_t=pw.this.sigma_t,
    resid=pw.this.resid,
    end_ts=pw.this.end_ts,  # Window end timestamp for join operation
)

logger.info("Joining GARCH/ARMA metrics back to original quotes using asof_join")

# Join GARCH/ARMA metrics back to original quotes using asof_join
# asof_join matches each quote with the most recent window that has ended
# This ensures we use the latest available GARCH/ARMA forecasts
# Use coalesce to handle cases where no matching window exists (defaults to safe values)
logger.debug("Performing asof_join: matching quotes with GARCH window end timestamps")
enriched_with_garch_join = quotes.asof_join(
    windowed_combined,
    quotes.ts_ms,  # Quote timestamp
    windowed_combined.end_ts,  # Window end timestamp
    quotes.symbol == windowed_combined.symbol,  # Match by symbol
    how=pw.JoinMode.LEFT,  # Keep all quotes even if no match
)

# Select and enrich quotes with GARCH/ARMA metrics
# Default values for missing metrics:
# - prev_close: 0.0 (no previous data)
# - ret: 0.0 (no return calculated)
# - resid: 0.0 (no residual)
# - sigma_t, sigma_forecast: 1e-3 (minimal volatility assumption)
# - arma_forecast: 0.0 (no forecast)
enriched_with_garch = enriched_with_garch_join.select(
    symbol=quotes.symbol,
    open=quotes.open,
    high=quotes.high,
    low=quotes.low,
    volume=quotes.volume,
    timestamp=quotes.timestamp,
    ts_ms=quotes.ts_ms,
    close=quotes.close,
    prev_close=pw.coalesce(windowed_combined.prev_close, 0.0),
    ret=pw.coalesce(windowed_combined.ret, 0.0),
    resid=pw.coalesce(windowed_combined.resid, 0.0),
    sigma_t=pw.coalesce(windowed_combined.sigma_t, 1e-3),
    sigma_forecast=pw.coalesce(windowed_combined.sigma_forecast, 1e-3),
    arma_forecast=pw.coalesce(windowed_combined.arma_forecast, 0.0),
)



# ============================================================================
# OPTIMIZATION 2: RSI (Relative Strength Index) Calculation + Signal Generation
# ============================================================================
# Combine RSI calculation and trading signal generation in a single pass
# to optimize performance and reduce computational overhead.

@pw.udf
def rsi_with_signal(price_tuples: tuple) -> tuple[float, int]:
    """
    Calculate RSI (Relative Strength Index) and generate trading signals.
    
    RSI is a momentum oscillator that measures the speed and magnitude of price changes.
    Values range from 0 to 100:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI = 50: Neutral
    
    Trading Signals:
    - rsi_timing = 2: Strong buy signal (rising RSI from oversold < 40)
    - rsi_timing = -2: Strong sell signal (falling RSI from overbought > 70)
    - rsi_timing = 0: No signal
    
    Args:
        price_tuples: List of (timestamp_ms, price) tuples, sorted by timestamp
    
    Returns:
        Tuple of (rsi, rsi_timing):
        - rsi: Current RSI value (0-100)
        - rsi_timing: Trading signal (-2, -1, 0, 1, 2)
    
    Note:
        Requires at least 14 periods for standard RSI calculation.
        Returns neutral RSI (50.0) and no signal (0) if insufficient data.
    """
    if len(price_tuples) < 14:
        logger.debug(f"rsi_with_signal: Insufficient data ({len(price_tuples)} < 14), returning neutral RSI")
        return (50.0, 0)

    prices = [p for (_, p) in price_tuples]

    # Calculate price changes (gains and losses) for RSI calculation
    # RSI uses average gains and losses over a specified period (typically 14)
    gains = []
    losses = []
    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gains.append(delta if delta > 0 else 0.0)  # Only positive changes
        losses.append(-delta if delta < 0 else 0.0)  # Only negative changes (as positive)

    # Use last 14 periods for RSI calculation (standard period)
    recent_gains = gains[-14:] if len(gains) >= 14 else gains
    recent_losses = losses[-14:] if len(losses) >= 14 else losses

    # Calculate average gain and average loss
    avg_gain = sum(recent_gains) / len(recent_gains) if recent_gains else 0.0
    avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0

    # Calculate RSI: RSI = 100 - (100 / (1 + RS))
    # where RS = Average Gain / Average Loss
    if avg_loss == 0:
        rsi = 100.0  # All gains, no losses (extreme overbought)
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    # Calculate RSI signal based on trend and divergence patterns
    # Need last 3 RSI values to detect patterns (rising/falling trends, reversals)
    rsi_history = []
    for end_idx in range(max(14, len(prices) - 2), len(prices) + 1):
        if end_idx < 14:
            continue
        start_idx = max(0, end_idx - 14)
        window_gains = gains[start_idx:end_idx]
        window_losses = losses[start_idx:end_idx]

        if window_gains and window_losses:
            w_avg_gain = sum(window_gains) / len(window_gains)
            w_avg_loss = sum(window_losses) / len(window_losses)
            if w_avg_loss == 0:
                rsi_history.append(100.0)
            else:
                w_rs = w_avg_gain / w_avg_loss
                rsi_history.append(100 - (100 / (1 + w_rs)))

    # Generate trading signals based on RSI patterns
    rsi_timing = 0
    if len(rsi_history) >= 3:
        r0, r1, r2 = rsi_history[-1], rsi_history[-2], rsi_history[-3]
        
        # Detect rising trend: RSI increasing over last 3 periods
        rising = (r0 > r1) and (r1 > r2)
        
        # Buy signal: Rising RSI from oversold levels (< 40)
        reversal_buy = min(rsi_history) < 40
        
        # Sell signal: Falling RSI from overbought levels (> 70)
        reversal_sell = (max(r0, r1, r2) > 70) and (r2 < r1 < r0)

        if rising and reversal_buy:
            rsi_timing = 2  # Strong buy signal
        elif reversal_sell:
            rsi_timing = -2  # Strong sell signal

    return (rsi, rsi_timing)


# Create sliding window for RSI calculation
# Window Configuration:
# - Duration: 17 periods (85 minutes) - 14 for RSI + 3 for signal pattern detection
# - Hop: 5 minutes - window slides every 5 minutes
# - Cutoff: 20 periods (100 minutes) - maximum delay before window is considered complete
#
# Note: This is a temporary optimization. Eventually this should be merged with
# the GARCH calculation window to reduce computational overhead.
logger.info("Creating sliding window for RSI calculation and signal generation")
logger.debug("RSI Window: 17 periods (85 min), hop: 5 min, cutoff: 100 min")

window_rsi_combined = enriched_with_garch.windowby(
    enriched_with_garch.ts_ms,
    window=pw.temporal.sliding(
        hop=5 * 60_000,  # 5 minutes
        duration=17 * 5 * 60_000,  # 17 periods = 85 minutes (14 for RSI + 3 for signal)
    ),
    instance=enriched_with_garch.symbol,  # Group by stock symbol
    behavior=pw.temporal.common_behavior(cutoff=20 * 5 * 60_000),  # 100 minutes max delay
).reduce(
    symbol=pw.this._pw_instance,
    price_tuples=pw.reducers.sorted_tuple(pw.make_tuple(pw.this.ts_ms, pw.this.close)),
    end_ts=pw.this._pw_window_end,
)

# Apply RSI calculation and extract timing signal
window_rsi_combined = window_rsi_combined.with_columns(
    rsi_result=rsi_with_signal(window_rsi_combined.price_tuples)
).with_columns(
    rsi_timing=pw.this.rsi_result[1]  # Extract signal (0, -2, or 2)
).select(
    symbol=pw.this.symbol,
    rsi_timing=pw.this.rsi_timing,
    end_ts=pw.this.end_ts,
)

# Join RSI signals back to enriched quotes
# Use coalesce to default to 0 (no signal) if no RSI data available
logger.debug("Joining RSI signals back to enriched quotes")
enriched_with_garch_rsi = enriched_with_garch.asof_join(
    window_rsi_combined,
    enriched_with_garch.ts_ms,
    window_rsi_combined.end_ts,
    enriched_with_garch.symbol == window_rsi_combined.symbol,
    how=pw.JoinMode.LEFT,
)

# Select all columns and add RSI timing signal
enriched_with_garch = enriched_with_garch_rsi.select(
    symbol=enriched_with_garch.symbol,
    open=enriched_with_garch.open,
    high=enriched_with_garch.high,
    low=enriched_with_garch.low,
    volume=enriched_with_garch.volume,
    timestamp=enriched_with_garch.timestamp,
    ts_ms=enriched_with_garch.ts_ms,
    close=enriched_with_garch.close,
    ret=enriched_with_garch.ret,
    resid=enriched_with_garch.resid,
    sigma_t=enriched_with_garch.sigma_t,
    sigma_forecast=enriched_with_garch.sigma_forecast,
    arma_forecast=enriched_with_garch.arma_forecast,
    rsi_timing=pw.coalesce(window_rsi_combined.rsi_timing, 0),  # Default: no signal
    prev_close=enriched_with_garch.prev_close,
)



# ============================================================================
# OPTIMIZATION 3: EMA (Exponential Moving Average) + MACD Calculation
# ============================================================================
# Combine all EMA calculations and MACD signal generation in a single window
# to optimize performance and reduce computational overhead.

@pw.udf
def calculate_ema_macd_signal(price_tuples: tuple) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculate multiple EMAs, MACD, and MACD signal line together.
    
    EMAs calculated:
    - EMA_12: 12-period EMA (fast)
    - EMA_26: 26-period EMA (slow, for MACD)
    - EMA_20: 20-period EMA (short-term trend)
    - EMA_50: 50-period EMA (medium-term trend)
    - EMA_200: 200-period EMA (long-term trend)
    
    MACD (Moving Average Convergence Divergence):
    - MACD = EMA_12 - EMA_26
    - Signal = 9-period EMA of MACD
    
    Args:
        price_tuples: List of (timestamp_ms, price) tuples, sorted by timestamp
    
    Returns:
        Tuple of (ema_12, ema_26, ema_20, ema_50, ema_200, macd, signal):
        - ema_12: 12-period exponential moving average
        - ema_26: 26-period exponential moving average
        - ema_20: 20-period exponential moving average
        - ema_50: 50-period exponential moving average
        - ema_200: 200-period exponential moving average
        - macd: MACD line (EMA_12 - EMA_26)
        - signal: MACD signal line (9-period EMA of MACD)
    
    Note:
        EMA smoothing factor alpha = 2 / (n + 1) where n is the period.
        Returns zeros if insufficient data is available.
    """
    prices = [p for (_, p) in price_tuples]

    if len(prices) == 0:
        logger.warning("calculate_ema_macd_signal: Empty price_tuples, returning default values")
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def calc_ema(prices, n, alpha):
        """
        Calculate Exponential Moving Average (EMA).
        
        EMA gives more weight to recent prices using a smoothing factor (alpha).
        Formula: EMA_t = alpha * Price_t + (1 - alpha) * EMA_{t-1}
        
        Args:
            prices: List of prices
            n: Period for EMA
            alpha: Smoothing factor (typically 2 / (n + 1))
        
        Returns:
            EMA value
        """
        if len(prices) == 0:
            return 0.0
        # Initialize with Simple Moving Average (SMA) for first n periods
        sma_len = min(len(prices), n)
        ema = sum(prices[:sma_len]) / sma_len
        # Apply EMA formula for remaining periods
        for i in range(sma_len, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema
        return ema

    # Calculate all required EMAs
    # Smoothing factors: alpha = 2 / (period + 1)
    ema_12 = calc_ema(prices, 12, 2 / 13)
    ema_26 = calc_ema(prices, 26, 2 / 27)
    ema_20 = calc_ema(prices, 20, 2 / 21)
    ema_50 = calc_ema(prices, 50, 2 / 51)
    ema_200 = calc_ema(prices, 200, 2 / 201)

    # Calculate MACD line: difference between fast and slow EMA
    macd = ema_12 - ema_26

    # Calculate MACD signal line (9-period EMA of MACD)
    # Need to calculate MACD history to compute signal line
    # Signal line requires MACD values over time, so we compute MACD for each point
    macd_history = []
    for end_idx in range(26, len(prices) + 1):  # Start from period 26 (minimum for EMA_26)
        window_prices = prices[:end_idx]
        w_ema12 = calc_ema(window_prices, 12, 2 / 13)
        w_ema26 = calc_ema(window_prices, 26, 2 / 27)
        macd_history.append(w_ema12 - w_ema26)

    # Signal line is 9-period EMA of MACD values
    signal = calc_ema(macd_history, 9, 2 / 10) if macd_history else 0.0

    return (ema_12, ema_26, ema_20, ema_50, ema_200, macd, signal)


# Create sliding window for EMA/MACD calculation
# Window Configuration:
# - Duration: 210 minutes - sufficient for EMA_200 (200 periods * 5 min = 1000 min, but we use shorter window)
#   Note: EMA_200 requires 200 data points, but we use a shorter window with buffer
# - Hop: 5 minutes - window slides every 5 minutes
# - Cutoff: 250 minutes - maximum delay before window is considered complete
logger.info("Creating sliding window for EMA and MACD calculation")
logger.debug("EMA Window: 210 minutes, hop: 5 min, cutoff: 250 min")

windowed_ema_all = enriched_with_garch.windowby(
    enriched_with_garch.ts_ms,
    window=pw.temporal.sliding(
        hop=5 * 60_000,  # 5 minutes
        duration=210 * 60_000,  # 210 minutes (42 periods) - buffer for EMA_200
    ),
    instance=enriched_with_garch.symbol,  # Group by stock symbol
    behavior=pw.temporal.common_behavior(cutoff=250 * 60_000),  # 250 minutes max delay
).reduce(
    symbol=pw.this._pw_instance,
    prices=pw.reducers.sorted_tuple(pw.make_tuple(pw.this.ts_ms, pw.this.close)),
    end_ts=pw.this._pw_window_end,
)

# Apply EMA/MACD calculation and extract individual metrics
windowed_ema_all = windowed_ema_all.with_columns(
    ema_result=calculate_ema_macd_signal(windowed_ema_all.prices)
).with_columns(
    # Extract individual EMA and MACD values from tuple result
    ema_12=pw.this.ema_result[0],
    ema_26=pw.this.ema_result[1],
    ema_20=pw.this.ema_result[2],
    ema_50=pw.this.ema_result[3],
    ema_200=pw.this.ema_result[4],
    macd=pw.this.ema_result[5],
    signal=pw.this.ema_result[6],
).select(
    symbol=pw.this.symbol,
    ema_12=pw.this.ema_12,
    ema_26=pw.this.ema_26,
    ema_20=pw.this.ema_20,
    ema_50=pw.this.ema_50,
    ema_200=pw.this.ema_200,
    macd=pw.this.macd,
    signal=pw.this.signal,
    end_ts=pw.this.end_ts,
)

# Join EMA/MACD metrics back to enriched quotes
# Use coalesce to default to 0.0 if no EMA data available
logger.debug("Joining EMA/MACD metrics back to enriched quotes")
enriched_final_join = enriched_with_garch.asof_join(
    windowed_ema_all,
    enriched_with_garch.ts_ms,
    windowed_ema_all.end_ts,
    enriched_with_garch.symbol == windowed_ema_all.symbol,
    how=pw.JoinMode.LEFT,
)

# Select all columns and add EMA/MACD metrics
enriched_final = enriched_final_join.select(
    symbol=enriched_with_garch.symbol,
    timestamp=enriched_with_garch.timestamp,
    ts_ms=enriched_with_garch.ts_ms,
    open=enriched_with_garch.open,
    high=enriched_with_garch.high,
    low=enriched_with_garch.low,
    close=enriched_with_garch.close,
    volume=enriched_with_garch.volume,
    ret=enriched_with_garch.ret,
    resid=enriched_with_garch.resid,
    sigma_t=enriched_with_garch.sigma_t,
    sigma_forecast=enriched_with_garch.sigma_forecast,
    arma_forecast=enriched_with_garch.arma_forecast,
    rsi_timing=enriched_with_garch.rsi_timing,
    prev_close=enriched_with_garch.prev_close,
    ema_12=pw.coalesce(windowed_ema_all.ema_12, 0.0),
    ema_26=pw.coalesce(windowed_ema_all.ema_26, 0.0),
    ema_20=pw.coalesce(windowed_ema_all.ema_20, 0.0),
    ema_50=pw.coalesce(windowed_ema_all.ema_50, 0.0),
    ema_200=pw.coalesce(windowed_ema_all.ema_200, 0.0),
    macd=pw.coalesce(windowed_ema_all.macd, 0.0),
    signal=pw.coalesce(windowed_ema_all.signal, 0.0),
)
logger.info("EMA/MACD enrichment completed")

# ============================================================================
# Derived Metrics Calculation
# ============================================================================
# Calculate additional trading indicators and filters based on EMA, MACD, and GARCH metrics
logger.info("Calculating derived trading metrics and filters")

enriched_final = enriched_final.with_columns(
    # MACD Histogram: difference between MACD line and signal line
    # Positive histogram = bullish momentum, Negative = bearish momentum
    histogram=enriched_final.macd - enriched_final.signal,
    
    # EMA Trend Filters: determine short-term trend direction
    # Trend up: EMA_20 > EMA_50 (bullish short-term trend)
    # Trend down: EMA_20 < EMA_50 (bearish short-term trend)
    ema_trend_filter_trend_up=enriched_final.ema_20 > enriched_final.ema_50,
    ema_trend_filter_trend_down=enriched_final.ema_20 < enriched_final.ema_50,
    
    # Long-term Bias: determine long-term trend direction
    # Trend up: Price > EMA_200 (bullish long-term trend)
    # Trend down: Price < EMA_200 (bearish long-term trend)
    long_term_bias_trend_up=enriched_final.close > enriched_final.ema_200,
    long_term_bias_trend_down=enriched_final.close < enriched_final.ema_200,
    
    # Risk-Adjusted Return: Sharpe-like ratio
    # Higher values indicate better risk-adjusted expected returns
    risk_adj_ret=pw.if_else(
        enriched_final.sigma_forecast > 0,
        enriched_final.arma_forecast / enriched_final.sigma_forecast,
        0.0
    ),
    
    # Simple directional signals based on ARMA forecast
    # Long signal: positive expected return
    # Short signal: negative expected return
    long_signal=(enriched_final.arma_forecast > 0),
    short_signal=(enriched_final.arma_forecast < 0),
    
    # Percentage change from previous period
    # Used for tracking price movements
    pct_change=pw.if_else(
        (enriched_final.prev_close > 0),
        ((enriched_final.close - enriched_final.prev_close) / enriched_final.prev_close) * 100,
        0.0
    )
)

# ============================================================================
# OPTIMIZATION 4: MACD Histogram Tracking and Signal Generation
# ============================================================================
# Track histogram changes to generate MACD trading signals
# MACD signals are based on histogram direction and position relative to zero

# Create small sliding window to track previous histogram value
# Window Configuration:
# - Duration: 10 minutes (2 periods) - just enough to get previous value
# - Hop: 5 minutes - window slides every 5 minutes
# - Cutoff: 15 minutes - maximum delay
logger.info("Creating window for MACD histogram tracking")
logger.debug("Histogram Window: 10 minutes, hop: 5 min, cutoff: 15 min")

windowed_histogram = enriched_final.windowby(
    enriched_final.ts_ms,
    window=pw.temporal.sliding(
        hop=5 * 60_000,  # 5 minutes
        duration=10 * 60_000,  # 10 minutes (2 periods)
    ),
    instance=enriched_final.symbol,  # Group by stock symbol
    behavior=pw.temporal.common_behavior(cutoff=15 * 60_000),  # 15 minutes max delay
).reduce(
    symbol=pw.this._pw_instance,
    histogram_values=pw.reducers.sorted_tuple(pw.make_tuple(pw.this.ts_ms, pw.this.histogram)),
    end_ts=pw.this._pw_window_end,
)


@pw.udf
def get_prev_histogram(values: tuple) -> float | None:
    """
    Extract previous period's histogram value from sorted tuple.
    
    Args:
        values: Sorted tuple of (timestamp_ms, histogram_value) pairs
    
    Returns:
        Previous histogram value (second-to-last) or None if insufficient data
    """
    if len(values) < 2:
        return None
    return values[-2][1]  # Return histogram value from second-to-last entry


# Extract previous histogram value
windowed_histogram = windowed_histogram.with_columns(
    histogram_prev=get_prev_histogram(windowed_histogram.histogram_values),
).select(
    symbol=pw.this.symbol,
    histogram_prev=pw.this.histogram_prev,
    end_ts=pw.this.end_ts,
)

# Join previous histogram value back to enriched data
logger.debug("Joining previous histogram values back to enriched quotes")
enriched_final_hist = enriched_final.asof_join(
    windowed_histogram,
    enriched_final.ts_ms,
    windowed_histogram.end_ts,
    enriched_final.symbol == windowed_histogram.symbol,
    how=pw.JoinMode.LEFT,
)

enriched_final = enriched_final_hist.select(
    *enriched_final,
    histogram_prev=windowed_histogram.histogram_prev,
)

# ============================================================================
# MACD Signal Classification
# ============================================================================
# Generate trading signals based on histogram position and direction
# Signal strength depends on:
# 1. Histogram position (above/below zero)
# 2. Histogram direction (growing/shrinking)
#
# Signal Values:
# - 2: Strong Buy (histogram > 0 and growing)
# - 1: Weak Buy (histogram > 0 but shrinking)
# - 0: No signal (neutral)
# - -1: Weak Sell (histogram < 0 but growing)
# - -2: Strong Sell (histogram < 0 and shrinking)
logger.info("Classifying MACD trading signals based on histogram")

# Determine histogram direction (growing or shrinking)
enriched_final = enriched_final.with_columns(
    histogram_growing=pw.if_else(
        pw.this.histogram_prev.is_not_none(),
        pw.this.histogram > pw.this.histogram_prev,
        False
    ),
    histogram_shrinking=pw.if_else(
        pw.this.histogram_prev.is_not_none(),
        pw.this.histogram < pw.this.histogram_prev,
        False
    ),
)

# Generate MACD signal based on histogram position and direction
enriched_final = enriched_final.with_columns(
    macd_signal=pw.if_else(
        (pw.this.histogram > 0) & pw.this.histogram_growing,
        2,  # Strong Buy: positive histogram and increasing momentum
        pw.if_else(
            (pw.this.histogram > 0) & pw.this.histogram_shrinking,
            1,  # Weak Buy: positive histogram but decreasing momentum
            pw.if_else(
                (pw.this.histogram < 0) & pw.this.histogram_shrinking,
                -2,  # Strong Sell: negative histogram and increasing negative momentum
                pw.if_else(
                    (pw.this.histogram < 0) & pw.this.histogram_growing,
                    -1,  # Weak Sell: negative histogram but decreasing negative momentum
                    0  # No signal: neutral or insufficient data
                )
            )
        )
    )
)
logger.info("MACD signal classification completed")



# ============================================================================
# Final Output Table Selection
# ============================================================================
# Select only the columns needed for downstream consumers
# This reduces data volume and focuses on key trading indicators
logger.info("Preparing final output table with selected columns")

final_table = enriched_final.select(
    symbol=enriched_final.symbol,
    timestamp=enriched_final.timestamp,
    ts_ms=enriched_final.ts_ms,
    close=enriched_final.close,
    sigma_forecast=enriched_final.sigma_forecast,
    arma_forecast=enriched_final.arma_forecast,
    ema_trend_filter_trend_up=enriched_final.ema_trend_filter_trend_up,
    ema_trend_filter_trend_down=enriched_final.ema_trend_filter_trend_down,
    long_term_bias_trend_up=enriched_final.long_term_bias_trend_up,
    long_term_bias_trend_down=enriched_final.long_term_bias_trend_down,
    macd_signal=enriched_final.macd_signal,
    risk_adj_ret=enriched_final.risk_adj_ret,
    long_signal=enriched_final.long_signal,
    short_signal=enriched_final.short_signal,
    rsi_timing=enriched_final.rsi_timing,
    pct_change=enriched_final.pct_change
)

# ============================================================================
# Kafka Output Configuration
# ============================================================================
# Write enriched technical analysis data to Kafka topics for downstream consumers

# Prepare timestamp output for synchronization/coordination purposes
# This is a separate topic that may be used by other services for timing
logger.info("Preparing timestamp output for Kafka topic 'stock_timestamp'")
timestamp_output = quotes.select(
    quotes.timestamp  # Just the timestamp string, no key
)

# Write timestamp to separate Kafka topic
# Format: raw (plain string) for simple timestamp forwarding
try:
    logger.info(f"Configuring Kafka writer for topic 'stock_timestamp' (broker: {kafka_broker})")
    pw.io.kafka.write(
        timestamp_output,
        {
            "bootstrap.servers": kafka_broker,
            "group.id": "pathway-group-timestamp",
        },
        topic_name="stock_timestamp",
        format="raw",  # Use "raw" format to send plain strings
    )
    logger.info("Kafka writer configured for 'stock_timestamp' topic")
except Exception as e:
    logger.error(f"Failed to configure Kafka writer for 'stock_timestamp': {str(e)}", exc_info=True)
    raise

# Write final enriched table with all technical indicators to Kafka
# This is the main output topic containing all calculated metrics
try:
    logger.info(f"Configuring Kafka writer for topic 'stock_calculation_table' (broker: {kafka_broker})")
    pw.io.kafka.write(
        final_table,
        rdkafka_settings={
            "bootstrap.servers": kafka_broker,
        },
        topic_name="stock_calculation_table",
        format="json",  # JSON format for structured data
    )
    logger.info("Kafka writer configured for 'stock_calculation_table' topic")
except Exception as e:
    logger.error(f"Failed to configure Kafka writer for 'stock_calculation_table': {str(e)}", exc_info=True)
    raise

# ============================================================================
# Start Pathway Pipeline
# ============================================================================
# Run the Pathway computation graph
# This will start processing data from Kafka input and writing to Kafka output
logger.info("Starting Pathway computation pipeline...")
logger.info("Service is now processing stock data and generating technical indicators")
try:
    pw.run()
except KeyboardInterrupt:
    logger.info("Pipeline interrupted by user")
    raise
except Exception as e:
    logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
    raise