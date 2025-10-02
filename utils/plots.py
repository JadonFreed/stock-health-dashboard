import plotly.graph_objects as go
import pandas as pd

def plot_price_with_indicators(hist):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"))
    if "SMA20" in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA20"], name="SMA20", line=dict(width=1)))
    if "SMA50" in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA50"], name="SMA50", line=dict(width=1)))
    if "SMA200" in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA200"], name="SMA200", line=dict(width=1)))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    return fig

# utils/plots.py

def plot_health_time_series(health_ts):
    """
    health_ts: DataFrame indexed by date with columns short_score,long_score,final_score
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    if "short_score" in health_ts.columns:
        fig.add_trace(go.Scatter(x=health_ts.index, y=health_ts["short_score"], name="Short-Term Health"))
    if "long_score" in health_ts.columns:
        fig.add_trace(go.Scatter(x=health_ts.index, y=health_ts["long_score"], name="Long-Term Health"))
    if "final_score" in health_ts.columns:
        fig.add_trace(go.Scatter(x=health_ts.index, y=health_ts["final_score"], name="Final Health", line=dict(width=3)))
    fig.update_layout(yaxis_title="Score (0-100)", yaxis=dict(range=[0, 100]), height=380)
    return fig

