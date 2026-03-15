import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow; fall back to sklearn if unavailable
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


def fetch_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical OHLCV data for a symbol."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


def get_stock_info(symbol: str) -> dict:
    """Return basic company info."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            "name":    info.get("longName", symbol),
            "sector":  info.get("sector", "N/A"),
            "country": info.get("country", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "country": "N/A",
                "market_cap": 0, "pe_ratio": None}


def _train_lstm(X_train, y_train, seq_len: int = 60):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
    return model


def _train_lr(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def get_prediction(symbol: str) -> dict:
    """
    Core prediction function.
    Returns current price, predicted next-day price, and BUY/SELL/HOLD signal.
    """
    df = fetch_stock_data(symbol)
    info = get_stock_info(symbol)
    prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    SEQ_LEN = 60
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if LSTM_AVAILABLE:
        model = _train_lstm(X_train, y_train, SEQ_LEN)
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    else:
        model = _train_lr(X_train, y_train)
        y_pred_scaled = model.predict(X_test.reshape(len(X_test), -1))

    # Metrics
    y_pred  = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae     = float(np.mean(np.abs(y_pred - y_actual)))
    accuracy = float(100 - (mae / np.mean(y_actual) * 100))

    # Predict next day
    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    if LSTM_AVAILABLE:
        next_scaled = model.predict(last_seq, verbose=0)[0, 0]
    else:
        next_scaled = model.predict(last_seq.reshape(1, -1))[0]

    next_price = float(scaler.inverse_transform([[next_scaled]])[0, 0])
    current    = float(prices[-1, 0])
    change_pct = (next_price - current) / current * 100

    if change_pct > 2:
        signal = "BUY"
    elif change_pct < -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Last 30 days chart data
    recent = df.tail(60)
    chart_dates  = recent.index.strftime('%Y-%m-%d').tolist()
    chart_prices = [round(p, 2) for p in recent['Close'].tolist()]

    return {
        "symbol":          symbol.upper(),
        "name":            info["name"],
        "sector":          info["sector"],
        "current_price":   round(current, 2),
        "predicted_price": round(next_price, 2),
        "change_pct":      round(change_pct, 2),
        "signal":          signal,
        "model_accuracy":  round(max(accuracy, 50), 1),
        "mae":             round(mae, 2),
        "chart_dates":     chart_dates,
        "chart_prices":    chart_prices,
        "model_used":      "LSTM" if LSTM_AVAILABLE else "LinearRegression",
    }
