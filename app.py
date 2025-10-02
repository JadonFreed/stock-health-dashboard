import streamlit as st
import plotly.graph_objs as go
from utils.data_fetch import load_config, get_stock_data, get_news_sentiment
from utils.health_calc import calculate_health

# ========= CONFIG =========
st.set_page_config(page_title="Stock Health Dashboard", layout="wide")
st.title("📈 Stock Health Dashboard")

config = load_config()

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()

if ticker:
    stock, hist = get_stock_data(ticker)
    sentiment = get_news_sentiment(ticker, config["ALPHA_VANTAGE_KEY"])
    health = calculate_health(hist, stock.info, sentiment)

    # ====== OUTPUT ======
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Indicators")
        st.write(f"**Short Term Trend:** {health['short_trend']}")
        st.write(f"**Short Score:** {health['short_score']}")
        st.write(f"**Long Score:** {health['long_score']}")
        st.write(f"**Final Health Index:** {health['final_score']:.2f}")
        st.write(f"P/E Ratio: {health['pe_ratio']}")
        st.write(f"Debt-to-Equity: {health['debt_to_equity']}")

    with col2:
        st.subheader("📉 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA100"], name="SMA100"))
        st.plotly_chart(fig, use_container_width=True)
