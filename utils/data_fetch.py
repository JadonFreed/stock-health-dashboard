import json
import os
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf

def load_config():
    cfg_path = "config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def get_stock_data(ticker, period="2y", interval="1d"):
    """
    Returns (yf.Ticker, hist_df)
    """
    t = yf.Ticker(ticker)
    # use try/except in case yfinance fails temporarily
    hist = t.history(period=period, interval=interval, auto_adjust=False)
    hist = hist.dropna(how="all")
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker}")
    hist = hist.sort_index()
    return t, hist

def get_fundamentals(ticker_obj):
    """
    Parse yfinance.info into a normalized dict with fields that health_calc expects.
    """
    info = {}
    try:
        raw = ticker_obj.info
    except Exception:
        raw = {}
    # common keys (may be None)
    info["trailingPE"] = raw.get("trailingPE")
    info["forwardPE"] = raw.get("forwardPE")
    info["priceToBook"] = raw.get("priceToBook") or raw.get("pbRatio")
    info["pegRatio"] = raw.get("pegRatio")
    info["returnOnEquity"] = raw.get("returnOnEquity")
    info["returnOnAssets"] = raw.get("returnOnAssets")
    info["debtToEquity"] = raw.get("debtToEquity")
    info["currentRatio"] = raw.get("currentRatio")
    # freeCashflow sometimes under different keys
    info["freeCashflow"] = raw.get("freeCashflow") or raw.get("freeCashFlow")
    info["marketCap"] = raw.get("marketCap")
    # try to get earnings/revenue history if present
    try:
        # df with 'Earnings' if available or quarterly/annual financials
        info["financials"] = {
            "financials": getattr(ticker_obj, "financials", None),
            "quarterly_financials": getattr(ticker_obj, "quarterly_financials", None),
            "earnings": getattr(ticker_obj, "earnings", None),
            "quarterly_earnings": getattr(ticker_obj, "quarterly_earnings", None)
        }
    except Exception:
        info["financials"] = {}
    return info

def get_news_yf(ticker):
    """
    Grab news from yfinance Ticker.news (fast, often contains title+link+providerPublishTime).
    Returns DataFrame with columns ['datetime','headline','source','summary' optional]
    """
    t = yf.Ticker(ticker)
    out = []
    try:
        raw = t.news
    except Exception:
        raw = []
    for item in raw:
        title = item.get("title") or item.get("link", "")
        src = item.get("publisher") or item.get("provider") or item.get("publisher")
        ts = None
        if item.get("providerPublishTime"):
            try:
                ts = datetime.fromtimestamp(int(item.get("providerPublishTime")))
            except Exception:
                ts = None
        # fallback: try 'time' or other key
        if not ts and item.get("time"):
            try:
                ts = datetime.fromtimestamp(int(item.get("time")))
            except Exception:
                ts = None
        out.append({
            "datetime": ts or datetime.utcnow(),
            "headline": title,
            "source": src or "yfinance"
        })
    if not out:
        return pd.DataFrame(columns=["datetime", "headline", "source"])
    df = pd.DataFrame(out)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df

def get_news_alpha_vantage(ticker, alpha_key):
    """
    Use Alpha Vantage NEWS_SENTIMENT if available. Returns same DataFrame schema.
    """
    if not alpha_key:
        return pd.DataFrame(columns=["datetime", "headline", "source"])
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        feed = data.get("feed", [])
    except Exception:
        feed = []
    rows = []
    for item in feed:
        ts = None
        try:
            ts = datetime.fromtimestamp(int(item.get("time_published")) / 1000) if item.get("time_published") else None
        except Exception:
            ts = None
        rows.append({
            "datetime": ts or datetime.utcnow(),
            "headline": item.get("title") or item.get("summary") or item.get("text") or "",
            "source": item.get("source") or "alpha_vantage"
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    return df

def get_news_combined(ticker, config):
    df_yf = get_news_yf(ticker)  # returns datetime, headline, source
    df_av = get_news_alpha_vantage(ticker, config.get("ALPHA_VANTAGE_KEY")) if config.get("ALPHA_VANTAGE_KEY") else pd.DataFrame()
    if df_av is None:
        df_av = pd.DataFrame()
    df = pd.concat([df_yf, df_av], ignore_index=True, sort=False).drop_duplicates(subset=["headline"])
    if "datetime" not in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    if "datetime" not in df.columns:
        # try index
        if isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
        else:
            df["datetime"] = pd.NaT
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "headline", "source"]]

def get_macro(config):
    """
    Minimal macro fetcher. Use FRED if key present (requires fredapi). Otherwise placeholder values.
    """
    fred_key = config.get("FRED_KEY")
    try:
        if fred_key:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            cpi = fred.get_series_latest_release('CPIAUCSL')  # monthly CPI
            fedfunds = fred.get_series_latest_release('FEDFUNDS')
            vix = fred.get_series_latest_release('VIXCLS')
            return {"CPI": float(cpi[-1]) if len(cpi) else None,
                    "FEDFUNDS": float(fedfunds[-1]) if len(fedfunds) else None,
                    "VIX": float(vix[-1]) if len(vix) else None}
    except Exception:
        pass
    # fallback / placeholders
    return {"CPI": None, "FEDFUNDS": None, "VIX": None}

def get_popular_tickers():
    # Default popular tickers; you can replace or extend this list.
    return ["AAPL","MSFT","AMZN","GOOGL","NVDA","TSLA","META","BRK-B","JPM","V"]
