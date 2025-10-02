import yfinance as yf
import requests
import json

def load_config():
    with open("config.json") as f:
        return json.load(f)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    return stock, hist

def get_news_sentiment(ticker, alpha_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_key}"
    news_data = requests.get(url).json()
    if "feed" in news_data:
        scores = [article["overall_sentiment_score"] for article in news_data["feed"]]
        return sum(scores) / len(scores) if scores else 0
    return 0
