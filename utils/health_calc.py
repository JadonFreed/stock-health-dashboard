def calculate_health(hist, stock_info, news_sentiment):
    # Short-term indicator
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA100"] = hist["Close"].rolling(100).mean()
    short_trend = "Bullish" if hist["SMA20"].iloc[-1] > hist["SMA100"].iloc[-1] else "Bearish"

    short_score = 0
    short_score += 1 if short_trend == "Bullish" else -1
    short_score += 1 if news_sentiment > 0 else -1

    # Long-term indicator
    pe_ratio = stock_info.get("trailingPE", None)
    debt_to_equity = stock_info.get("debtToEquity", None)

    long_score = 0
    long_score += 1 if pe_ratio and pe_ratio < 25 else -1
    long_score += 1 if debt_to_equity and debt_to_equity < 100 else -1
    long_score += 1 if hist["Close"].iloc[-1] > hist["SMA100"].iloc[-1] else -1

    # Final health
    final_score = 0.6 * short_score + 0.4 * long_score

    return {
        "short_trend": short_trend,
        "short_score": short_score,
        "long_score": long_score,
        "final_score": final_score,
        "pe_ratio": pe_ratio,
        "debt_to_equity": debt_to_equity
    }
