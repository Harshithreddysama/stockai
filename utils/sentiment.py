import os
import requests
from functools import lru_cache

# Try FinBERT; fall back to a simple keyword approach if torch unavailable
try:
    from transformers import pipeline as hf_pipeline
    _sentiment_pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )
    FINBERT_AVAILABLE = True
except Exception:
    FINBERT_AVAILABLE = False

POSITIVE_WORDS = {"surge", "gain", "profit", "beat", "rally", "record",
                  "growth", "strong", "up", "rise", "high", "bullish",
                  "upgrade", "positive", "boost", "soar"}
NEGATIVE_WORDS = {"fall", "drop", "loss", "miss", "crash", "down",
                  "weak", "bearish", "cut", "risk", "concern",
                  "decline", "negative", "downgrade", "sell"}


def _keyword_sentiment(text: str) -> str:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


def _analyze_headlines(headlines: list[str]) -> list[dict]:
    if not headlines:
        return []
    if FINBERT_AVAILABLE:
        try:
            results = _sentiment_pipe(headlines[:10])
            return [{"label": r["label"], "score": r["score"]} for r in results]
        except Exception:
            pass
    return [{"label": _keyword_sentiment(h), "score": 0.6} for h in headlines]


def get_news_sentiment(symbol: str, company_name: str = "") -> dict:
    """
    Fetch latest news for a stock and return sentiment analysis.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    query   = company_name if company_name else symbol
    headlines = []

    if api_key and api_key != "your_newsapi_key_here":
        try:
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={query}&language=en&pageSize=15&sortBy=publishedAt"
                f"&apiKey={api_key}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                headlines = [a["title"] for a in articles if a.get("title")]
        except Exception:
            pass

    # Fallback: use mock headlines if no API key or request failed
    if not headlines:
        headlines = [
            f"{symbol} reports quarterly earnings results",
            f"Analysts update price target for {symbol}",
            f"Market outlook: {symbol} performance review",
        ]

    results = _analyze_headlines(headlines[:10])

    if not results:
        return {"score": 0.0, "label": "NEUTRAL", "headlines": headlines[:3],
                "positive": 0, "negative": 0, "neutral": 1}

    pos = sum(1 for r in results if r["label"] == "positive")
    neg = sum(1 for r in results if r["label"] == "negative")
    neu = sum(1 for r in results if r["label"] == "neutral")
    score = (pos - neg) / len(results)

    label = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"

    return {
        "score":     round(score, 2),
        "label":     label,
        "headlines": headlines[:5],
        "positive":  pos,
        "negative":  neg,
        "neutral":   neu,
    }


def combine_signals(ml_signal: str, sentiment_label: str,
                    sentiment_score: float, change_pct: float) -> str:
    """
    Combine ML model signal with news sentiment into a final signal.
    Conservative: if sentiment contradicts model, downgrade to HOLD.
    """
    if ml_signal == "BUY" and sentiment_label == "BEARISH" and sentiment_score < -0.3:
        return "HOLD"
    if ml_signal == "SELL" and sentiment_label == "BULLISH" and sentiment_score > 0.3:
        return "HOLD"
    if ml_signal == "HOLD" and sentiment_label == "BULLISH" and change_pct > 0:
        return "BUY"
    if ml_signal == "HOLD" and sentiment_label == "BEARISH" and change_pct < 0:
        return "SELL"
    return ml_signal
