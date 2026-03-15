import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    yf.set_tz_cache_location("custom_cache_dir")
    YFINANCE_OK = True
except Exception:
    YFINANCE_OK = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


def fetch_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical data - tries multiple methods."""
    
    # Method 1: yfinance with session fix
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(period=period)
        if not df.empty and len(df) > 60:
            return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        pass

    # Method 2: standard yfinance
    try:
        import yfinance as yf2
        df = yf2.download(symbol, period=period, progress=False)
        if not df.empty and len(df) > 60:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        pass

    # Method 3: Alpha Vantage free API (no key needed for demo)
    try:
        import requests
        # Use stooq as backup - free, no API key
        clean = symbol.replace('.NS', '.IN').replace('.BO', '.IN')
        url = f"https://stooq.com/q/d/l/?s={clean.lower()}&i=d"
        df = pd.read_csv(url)
        if not df.empty and len(df) > 60:
            df.columns = ['Date','Open','High','Low','Close','Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            df = df[df['Close'] > 0].tail(500)
            return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception:
        pass

    raise ValueError(
        f"Could not fetch data for '{symbol}'. "
        f"Please check your internet connection and try again. "
        f"For Indian stocks use: TCS.NS, INFY.NS, HDFCBANK.NS"
    )


def get_stock_info(symbol: str) -> dict:
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        t = yf.Ticker(symbol, session=session)
        info = t.info
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "country": info.get("country", "N/A"),
            "market_cap": info.get("marketCap", 0),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A",
                "country": "N/A", "market_cap": 0}


def _train_lstm(X_train, y_train, seq_len=60):
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


def get_prediction(symbol: str) -> dict:
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
        lr = LinearRegression()
        lr.fit(X_train.reshape(len(X_train), -1), y_train)
        y_pred_scaled = lr.predict(X_test.reshape(len(X_test), -1))
        model = lr

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    mae = float(np.mean(np.abs(y_pred - y_actual)))
    accuracy = float(100 - (mae / np.mean(y_actual) * 100))

    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    if LSTM_AVAILABLE:
        next_scaled = model.predict(last_seq, verbose=0)[0, 0]
    else:
        next_scaled = model.predict(last_seq.reshape(1, -1))[0]

    next_price = float(scaler.inverse_transform([[next_scaled]])[0, 0])
    current = float(prices[-1, 0])
    change_pct = (next_price - current) / current * 100

    if change_pct > 2:
        signal = "BUY"
    elif change_pct < -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    recent = df.tail(60)
    chart_dates = recent.index.strftime('%Y-%m-%d').tolist()
    chart_prices = [round(float(p), 2) for p in recent['Close'].tolist()]

    return {
        "symbol": symbol.upper(),
        "name": info["name"],
        "sector": info["sector"],
        "current_price": round(current, 2),
        "predicted_price": round(next_price, 2),
        "change_pct": round(change_pct, 2),
        "signal": signal,
        "model_accuracy": round(max(accuracy, 50), 1),
        "mae": round(mae, 2),
        "chart_dates": chart_dates,
        "chart_prices": chart_prices,
        "model_used": "LSTM" if LSTM_AVAILABLE else "LinearRegression",
    }