# 📈 Stock Health Dashboard

An interactive dashboard for evaluating **short-term, long-term, and overall stock health**, with live data + sentiment.

## 🚀 Features
- Live prices & indicators (Yahoo Finance / yfinance)
- Fundamental ratios
- News sentiment (Alpha Vantage API)
- Short-term, long-term, and final health index
- Interactive dashboard (Streamlit + Plotly)

## ⚙️ Setup
```bash
git clone https://github.com/yourusername/stock-health-dashboard.git
cd stock-health-dashboard
pip install -r requirements.txt
cp config_example.json config.json   # Add your API keys
streamlit run app.py
