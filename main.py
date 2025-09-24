import requests
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# 🔑 API Keys
BINANCE_API_KEY = "0TlWBhGYxBwQiJ6dcea3JfPzUW4ri2YsuDEcS1jVvGsNl1kLnbawFr8LBe4Ts6Sk"
NEWS_API_KEY = "ce3facc9bafb4f26ab62a47e3b073c99"

# API Endpoints
BINANCE_ENDPOINT = "https://api.binance.com/api/v3/klines"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

class SymbolRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    limit: int = 150

# ===== Indicators =====
def calculate_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period if len(deltas[deltas < 0]) > 0 else 0.0001
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_ema(closes, period=9):
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(closes, weights, mode='full')[:len(closes)]
    ema[:period] = ema[period]
    return ema

def calculate_macd(closes):
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    macd = ema12 - ema26
    signal = calculate_ema(macd, 9)
    hist = macd - signal
    return round(macd[-1], 4), round(signal[-1], 4), round(hist[-1], 4), macd, signal

def find_support_resistance(highs, lows, closes, levels=3):
    prices = np.concatenate([highs, lows, closes])
    prices = np.sort(prices)
    step = len(prices) // levels
    supports = [round(prices[i], 2) for i in range(step, len(prices), step)][:levels]
    resistances = [round(prices[-i], 2) for i in range(1, levels+1)]
    return sorted(supports), sorted(resistances)

@app.post("/analyze")
def analyze_market(req: SymbolRequest):
    # 1️⃣ Binance Data
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    params = {"symbol": req.symbol, "interval": req.interval, "limit": req.limit}
    candles = requests.get(BINANCE_ENDPOINT, headers=headers, params=params).json()
    
    closes = np.array([float(c[4]) for c in candles])   
    highs = np.array([float(c[2]) for c in candles])
    lows = np.array([float(c[3]) for c in candles])

    last_close = closes[-1]

    # 2️⃣ Indicators
    rsi = calculate_rsi(closes)
    macd, signal, hist, macd_series, signal_series = calculate_macd(closes)
    ema9 = calculate_ema(closes, 9)[-1]
    ema21 = calculate_ema(closes, 21)[-1]

    # Trend Detection
    if ema9 > ema21:
        trend = "Bullish"
    elif ema9 < ema21:
        trend = "Bearish"
    else:
        trend = "Sideways"

    # Entry, TP, SL
    entry = round(last_close, 2)
    tp = round(entry * 1.01, 2)
    sl = round(entry * 0.99, 2)

    # Success Probability
    if trend == "Bullish" and rsi < 70 and macd > signal:
        success_prob = "High"
    elif trend == "Bearish" and rsi > 30 and macd < signal:
        success_prob = "Medium"
    else:
        success_prob = "Low"

    # 3️⃣ Support & Resistance
    supports, resistances = find_support_resistance(highs, lows, closes)

    # 4️⃣ Signals
    signals = []
    if rsi > 70:
        signals.append("⚠️ RSI Overbought → احتمال انعكاس هابط (بيع)")
    elif rsi < 30:
        signals.append("⚠️ RSI Oversold → احتمال انعكاس صاعد (شراء)")

    if macd_series[-2] < signal_series[-2] and macd_series[-1] > signal_series[-1]:
        signals.append("✅ MACD Bullish Cross → إشارة شراء")
    elif macd_series[-2] > signal_series[-2] and macd_series[-1] < signal_series[-1]:
        signals.append("❌ MACD Bearish Cross → إشارة بيع")

    if not signals:
        signals.append("ℹ️ لا توجد إشارات قوية حالياً")

    # 5️⃣ News Data
    news_params = {
        "q": "crypto",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 3
    }
    news_data = requests.get(NEWS_ENDPOINT, params=news_params).json()
    headlines = [article["title"] for article in news_data.get("articles", [])]

    # 6️⃣ Output
    return {
        "symbol": req.symbol,
        "interval": req.interval,
        "last_price": entry,
        "trend": trend,
        "entry": entry,
        "take_profit": tp,
        "stop_loss": sl,
        "RSI": rsi,
        "MACD": macd,
        "Signal": signal,
        "Histogram": hist,
        "EMA9": round(float(ema9), 2),
        "EMA21": round(float(ema21), 2),
        "Supports": supports,
        "Resistances": resistances,
        "success_probability": success_prob,
        "trade_signals": signals,
        "recent_news": headlines
  }
