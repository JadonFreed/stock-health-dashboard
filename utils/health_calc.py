import math
import numpy as np
import pandas as pd
from datetime import timedelta

# ta library components
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# -------------------------
# helpers: scoring utilities
# -------------------------
def _safe_div(a, b):
    try:
        return a / b
    except Exception:
        return np.nan

def _scale_0_100(x, lo, hi, inverse=False):
    """
    Linearly scale x to 0..100 given lo..hi where:
      - if inverse=False: lo->0, hi->100
      - if inverse=True: lo->100, hi->0 (useful when lower is better)
    If x is None/NaN: return 50 (neutral)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 50
    if lo == hi:
        return 50
    if inverse:
        if x <= lo: return 100
        if x >= hi: return 0
        return int(100 * (hi - x) / (hi - lo))
    else:
        if x <= lo: return 0
        if x >= hi: return 100
        return int(100 * (x - lo) / (hi - lo))

# -------------------------
# technical indicators
# -------------------------
def add_technical_indicators(df):
    """
    Adds RSI, MACD, Bollinger bands, ATR, ADX, OBV, VWAP to df inplace and returns df.
    Expects columns: High, Low, Close, Volume
    """
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain Close column")

    # RSI
    try:
        df["rsi_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
    except Exception:
        df["rsi_14"] = np.nan

    # MACD
    try:
        macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception:
        df["macd"] = df["macd_signal"] = df["macd_diff"] = np.nan

    # Bollinger
    try:
        bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["bb_h"] = bb.bollinger_hband()
        df["bb_l"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_h"] - df["bb_l"]) / df["Close"]
    except Exception:
        df["bb_h"] = df["bb_l"] = df["bb_width"] = np.nan

    # ATR
    try:
        atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["atr"] = atr.average_true_range()
    except Exception:
        df["atr"] = np.nan

    # ADX
    try:
        adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
        df["adx"] = adx.adx()
    except Exception:
        df["adx"] = np.nan

    # OBV
    try:
        obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
        df["obv"] = obv.on_balance_volume()
    except Exception:
        df["obv"] = np.nan

    # VWAP (rolling window = 14)
    try:
        vwap = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=14)
        df["vwap"] = vwap.volume_weighted_average_price()
    except Exception:
        df["vwap"] = np.nan

    return df

# -------------------------
# technical scoring
# -------------------------
def technical_score_for_window(df_window):
    """
    Input: the most recent window (DataFrame). Returns a 0-100 technical score.
    Combines: momentum (MACD, RSI), trend (SMA relationships), volume, volatility penalty
    """
    if df_window.empty:
        return 50
    last = df_window.iloc[-1]
    # RSI score: prefer RSI in 45-70 (momentum), penalize >80 (overbought)
    rsi = float(last.get("rsi_14", np.nan) or np.nan)
    if np.isnan(rsi): rsi_score = 50
    else:
        # map 0..100 into 0..100 but slightly prefer >50
        if rsi < 30:
            rsi_score = 30
        elif rsi > 80:
            rsi_score = 30
        else:
            # linear 30..100 between 30..80
            rsi_score = int(30 + (rsi - 30) * (70 / 50))

    # MACD diff percentile in window
    macd_vals = df_window["macd_diff"].dropna()
    if len(macd_vals) < 3 or macd_vals.std() == 0:
        macd_score = 50
    else:
        recent = last.get("macd_diff", 0.0)
        # normalize to percentile
        macd_score = int(50 + 50 * ( (recent - macd_vals.min()) / (macd_vals.max() - macd_vals.min()) - 0.5))

    # Trend: SMA20 vs SMA50 vs SMA200 (we assume these columns present or compute loosely)
    close = df_window["Close"]
    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

    trend_score = 50
    if not np.isnan(sma20) and not np.isnan(sma50):
        if sma20 > sma50:
            trend_score += 20
    if not np.isnan(sma200) and close.iloc[-1] > sma200:
        trend_score += 15

    # Volume: last volume vs 20-day avg
    vol = float(last.get("Volume", 0) or 0)
    vol_avg = float(df_window["Volume"].rolling(20).mean().iloc[-1]) if len(df_window) >= 20 else np.nan
    vol_score = 50
    if not np.isnan(vol_avg) and vol_avg > 0:
        ratio = vol / vol_avg
        # reward higher-than-average volume on up-days
        vol_score = int(np.clip(50 + (ratio - 1) * 50, 0, 100))

    # Volatility penalty: higher ATR/price ratio reduces score
    atr = float(last.get("atr", np.nan) or np.nan)
    vol_penalty = 0
    if not np.isnan(atr):
        pct_atr = _safe_div(atr, close.iloc[-1])
        vol_penalty = int(np.clip(pct_atr * 300, 0, 50))  # bigger percent ATR => larger penalty

    # Combine weighted
    combined = (0.28 * rsi_score +
                0.28 * macd_score +
                0.22 * trend_score +
                0.12 * vol_score) - vol_penalty
    combined = int(np.clip(combined, 0, 100))
    return combined

# -------------------------
# fundamentals scoring
# -------------------------
def score_fundamentals(fund):
    """
    fund: dict from data_fetch.get_fundamentals
    Returns 0-100 fundamental score.
    Uses: PE (lower within reasonable), PB (lower), ROE, ROA, debtToEquity (lower), freeCashflow positive
    """
    # unpack safely
    pe = fund.get("trailingPE")
    pb = fund.get("priceToBook")
    roe = fund.get("returnOnEquity")
    roa = fund.get("returnOnAssets")
    dte = fund.get("debtToEquity")
    fcf = fund.get("freeCashflow")

    # Scores per metric (0..100)
    # PE: ideal <= 25 (score 100), 25..60 -> decreasing, >60 -> low
    if pe is None or (isinstance(pe, float) and math.isnan(pe)):
        pe_score = 50
    elif pe <= 25:
        pe_score = int(75 + max(0, (25 - pe) / 25 * 25))  # 75-100
    elif pe <= 60:
        pe_score = int(75 - (pe - 25) / 35 * 60)  # roughly 75->15
    else:
        pe_score = 10

    # PB: ideal <=5
    pb_score = _scale_0_100(pb, 0, 5, inverse=False) if pb is not None else 50

    # ROE: higher is better; typical good >15
    roe_score = _scale_0_100(roe*100 if roe is not None and abs(roe) < 2 else (roe if roe is not None else None), 0, 25) if roe is not None else 50
    # if roe came as 0.18 (18%) the conversion above sets appropriate scale.

    # ROA:
    roa_score = _scale_0_100(roa*100 if roa is not None and abs(roa) < 2 else (roa if roa is not None else None), 0, 15) if roa is not None else 50

    # Debt: lower is better; ideal <1
    debt_score = _scale_0_100(dte, 0, 2, inverse=True) if dte is not None else 50

    # FCF: presence & positive better
    fcf_score = 80 if (fcf is not None and fcf > 0) else 30 if (fcf is not None and fcf <= 0) else 50

    # combine with weights
    final = (0.18 * pe_score +
             0.16 * pb_score +
             0.22 * roe_score +
             0.12 * roa_score +
             0.16 * debt_score +
             0.16 * fcf_score)
    final = int(np.clip(final, 0, 100))
    return final

# -------------------------
# sentiment scoring
# -------------------------
def sentiment_score_from_series(series_compound):
    """
    Input: iterable of VADER/compound sentiment scores (-1..1)
    Returns 0-100 scaled score and mean sentiment
    """
    arr = np.array(series_compound)
    if len(arr) == 0:
        return 50, 0.0
    mean = float(np.nanmean(arr))
    scaled = int(np.clip((mean + 1) / 2 * 100, 0, 100))
    return scaled, mean

# -------------------------
# full health calculations
# -------------------------
def compute_scores_for_date(df, fundamentals_dict, news_df, date, short_window=20, news_window_days=30):
    """
    Compute scaled scores (0-100) as of `date`.
    Returns dict with keys:
      date, short_score, long_score, final_score, technical_short, fundamental_score, sentiment_score, avg_sentiment
    """
    # normalize date
    date = pd.to_datetime(date).tz_localize(None) if not pd.isna(date) else date

    # normalize news datetime column to 'datetime' and tz-naive
    if news_df is None:
        news_df = pd.DataFrame(columns=["datetime", "headline", "sentiment"])
    else:
        news_df = news_df.copy()
        # accept many possible timestamp column names
        if "datetime" in news_df.columns:
            news_df["datetime"] = pd.to_datetime(news_df["datetime"], errors="coerce").dt.tz_localize(None)
        elif "date" in news_df.columns:
            news_df["datetime"] = pd.to_datetime(news_df["date"], errors="coerce").dt.tz_localize(None)
        elif "publishedAt" in news_df.columns:
            news_df["datetime"] = pd.to_datetime(news_df["publishedAt"], errors="coerce").dt.tz_localize(None)
        elif isinstance(news_df.index, pd.DatetimeIndex):
            news_df["datetime"] = news_df.index.tz_localize(None)
        else:
            news_df["datetime"] = pd.NaT

    # slice historical df up to date
    hist = df[df.index <= date].copy()
    if hist.empty:
        return {
            "date": date,
            "short_score": np.nan,
            "long_score": np.nan,
            "final_score": np.nan,
            "technical_short": np.nan,
            "fundamental_score": np.nan,
            "sentiment_score": np.nan,
            "avg_sentiment": np.nan
        }

    # ensure technical indicators exist
    hist = add_technical_indicators(hist)

    # Short-term technical score (0-100)
    window = hist.tail(max(short_window, 30))
    technical_short = technical_score_for_window(window)  # expects 0..100

    # Sentiment: use headlines within news_window_days up to date
    since = date - timedelta(days=news_window_days)
    recent_news = news_df[(news_df["datetime"] <= date) & (news_df["datetime"] >= since)]
    # make sure sentiment column exists and is numeric
    if "sentiment" in recent_news.columns:
        compounds = pd.to_numeric(recent_news["sentiment"], errors="coerce").dropna().tolist()
    elif "overall_sentiment_score" in recent_news.columns:
        compounds = pd.to_numeric(recent_news["overall_sentiment_score"], errors="coerce").dropna().tolist()
    else:
        compounds = []

    sentiment_score, avg_sentiment = sentiment_score_from_series(compounds)  # 0..100, mean (-1..1)

    # short term health (weights)
    short_health = int(np.clip(0.6 * technical_short + 0.4 * sentiment_score, 0, 100))

    # fundamental score (0-100)
    fundamental_score = score_fundamentals(fundament_dict=fundamentals_dict) if fundamentals_dict is not None else 50
    # long trend: SMA200 check
    close = hist["Close"]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    long_trend_score = 70 if (not np.isnan(sma200) and close.iloc[-1] > sma200) else 40

    macro_score = 50  # placeholder, can be enhanced
    long_health = int(np.clip(0.6 * fundamental_score + 0.3 * long_trend_score + 0.1 * macro_score, 0, 100))

    final_score = int(np.clip(0.5 * short_health + 0.5 * long_health, 0, 100))

    return {
        "date": date,
        "short_score": short_health,
        "long_score": long_health,
        "final_score": final_score,
        "technical_short": technical_short,
        "fundamental_score": int(fundamental_score),
        "sentiment_score": int(sentiment_score),
        "avg_sentiment": float(avg_sentiment)
    }


def compute_health_time_series(df, fundamentals_dict, news_df, start_idx=None, short_window=20, news_window_days=30):
    """
    Compute health scores for each date in df (or from start_idx).
    Returns DataFrame indexed by date with columns: short_score, long_score, final_score, technical_short, fundamental_score, sentiment_score, avg_sentiment
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)

    if start_idx is not None:
        start_idx = pd.to_datetime(start_idx).tz_localize(None)
        df = df[df.index >= start_idx]

    rows = []
    for date in df.index:
        try:
            row = compute_scores_for_date(df, fundamentals_dict, news_df, date, short_window=short_window, news_window_days=news_window_days)
            rows.append(row)
        except Exception:
            # skip bad dates but continue
            continue

    if not rows:
        return pd.DataFrame(columns=["short_score", "long_score", "final_score"]).set_index(pd.Index([], name="date"))

    res = pd.DataFrame(rows).set_index("date")
    # ensure numeric types
    for col in ["short_score", "long_score", "final_score", "technical_short", "fundamental_score", "sentiment_score", "avg_sentiment"]:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce")
    return res
