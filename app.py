import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# utils
from utils.data_fetch import load_config, get_stock_data, get_fundamentals, get_news_combined, get_popular_tickers
from utils.health_calc import add_technical_indicators, compute_health_time_series, compute_scores_for_date, score_fundamentals
from utils.plots import plot_price_with_indicators, plot_health_time_series

st.set_page_config(page_title="Stock Health Dashboard", layout="wide")
st.title("📈 Stock Health Dashboard — Advanced")

config = load_config()

# Sidebar controls
st.sidebar.header("Controls")
popular = get_popular_tickers()
tickers = st.sidebar.multiselect("Pick tickers (or type):", popular, default=["AAPL"])
start_date = st.sidebar.date_input("Start date", value=(datetime.utcnow() - timedelta(days=365)).date())
end_date = st.sidebar.date_input("End date", value=datetime.utcnow().date())
analyze_btn = st.sidebar.button("Analyze selected tickers")

# risk tolerance slider used in recommendation step
risk_tol = st.sidebar.selectbox("Risk tolerance", ["Conservative", "Balanced", "Aggressive"])

# quick ticker search (single ticker)
single_ticker = st.sidebar.text_input("Quick single ticker", value="AAPL").upper()

# helper: compute and display for a single ticker when requested
def analyze_ticker(ticker, start_date, end_date):
    try:
        ticker_obj, hist = get_stock_data(ticker, period="2y")
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return None

    # clip to selected date range
    hist = hist[(hist.index.date >= start_date) & (hist.index.date <= end_date)].copy()
    if hist.empty:
        st.warning(f"No price data in the selected date range for {ticker}")
        return None

    # add rolling SMAs for visualization convenience
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    hist["SMA200"] = hist["Close"].rolling(200).mean()

    hist = add_technical_indicators(hist)

    fundamentals = get_fundamentals(ticker_obj)
    news_df = get_news_combined(ticker, config)
    # compute sentiment column with VADER by default if not present (health_calc expects it)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        news_df["sentiment"] = news_df["headline"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])
    except Exception:
        news_df["sentiment"] = 0.0

    # compute health time series (this may take a few seconds)
    health_ts = compute_health_time_series(hist, fundamentals, news_df, start_idx=hist.index.min())

    # final ranking/score for latest date
    latest = health_ts.iloc[-1].to_dict() if not health_ts.empty else {}
    latest.update({"fundamental_score": score_fundamentals(fundamentals)})

    return {
        "ticker": ticker,
        "hist": hist,
        "fundamentals": fundamentals,
        "news": news_df,
        "health_ts": health_ts,
        "latest": latest
    }

# Single quick analyze
if st.sidebar.button("Analyze quick ticker"):
    res = analyze_ticker(single_ticker, start_date, end_date)
    if res:
        st.experimental_set_query_params(ticker=single_ticker)
        st.write(f"### {single_ticker} — summary")
        st.write("Final score:", res["latest"].get("final_score"))
        st.plotly_chart(plot_price_with_indicators(res["hist"]), use_container_width=True)
        st.plotly_chart(plot_health_time_series(res["health_ts"]), use_container_width=True)
        st.dataframe(res["fundamentals"])

# Batch analyze when Analyze selected tickers pressed
if analyze_btn and tickers:
    results = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        progress.progress(int((i+1)/len(tickers) * 100))
        r = analyze_ticker(t, start_date, end_date)
        if r:
            results.append(r)
    progress.empty()

    # show recommendations table: sort by final_score desc
    rec_rows = []
    for r in results:
        latest = r["latest"]
        rec_rows.append({
            "ticker": r["ticker"],
            "final_score": latest.get("final_score", np.nan),
            "short_score": latest.get("short_score", np.nan),
            "long_score": latest.get("long_score", np.nan),
            "fundamental_score": latest.get("fundamental_score", np.nan),
            "avg_sentiment": latest.get("avg_sentiment", np.nan)
        })
    rec_df = pd.DataFrame(rec_rows).set_index("ticker").sort_values("final_score", ascending=False)
    st.subheader("Recommendations (sorted by Final Health Index)")
    st.dataframe(rec_df)

    # simple allocation suggestion: higher score => higher weight, adjusted by risk tolerance
    def allocation_from_score(score, risk):
        # baseline weight proportional to (score/100)
        base = max(0, score)
        if risk == "Conservative":
            # favor fundamentals: reduce allocation for high short-term-only scores
            return base * 0.8
        if risk == "Balanced":
            return base
        return base * 1.2

    alloc = []
    for tck, row in rec_df.iterrows():
        weight = allocation_from_score(row["final_score"], risk_tol)
        alloc.append({"ticker": tck, "score": row["final_score"], "weight_raw": weight})
    alloc_df = pd.DataFrame(alloc).set_index("ticker")
    if not alloc_df.empty:
        # normalize to 100%
        s = alloc_df["weight_raw"].sum()
        if s > 0:
            alloc_df["weight_pct"] = (alloc_df["weight_raw"] / s * 100).round(1)
        else:
            alloc_df["weight_pct"] = 0.0
        st.subheader("Suggested Allocation (rule-based)")
        st.dataframe(alloc_df[["score","weight_pct"]])

    # allow drill-down into any ticker
    st.markdown("### Drill down")
    select_drill = st.selectbox("Pick a ticker to drill into", rec_df.index.tolist())
    drill = next((r for r in results if r["ticker"] == select_drill), None)
    if drill:
        st.plotly_chart(plot_price_with_indicators(drill["hist"]), use_container_width=True)
        st.plotly_chart(plot_health_time_series(drill["health_ts"]), use_container_width=True)
        st.subheader("Latest metrics")
        st.write(drill["latest"])
        st.subheader("Recent headlines")
        st.dataframe(drill["news"].tail(10))

# footer / tips
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: results depend heavily on available fundamental fields and news coverage. For more accurate sentiment you can install optional FinBERT dependencies (transformers + torch) and configure the app to use it.")
