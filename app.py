# app.py
import streamlit as st
import yfinance as yf
from gnewsclient import gnewsclient
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import backtrader as bt
from pymongo import MongoClient
from alpha_vantage.timeseries import TimeSeries
import os
import time
import ta  # technical analysis helper

# ---------------------------
# Config / Constants
# ---------------------------
st.set_page_config(layout="wide", page_title="NextTick - Stock + News Prediction", initial_sidebar_state="expanded")

# Sidebar - user settings
st.sidebar.title("NextTick Settings")
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"PyTorch device: **{MODEL_DEVICE}**")

ALPHA_VANTAGE_KEY = st.sidebar.text_input("Alpha Vantage API Key (optional)", value="", help="Optional fallback data source")
MONGO_URI = st.sidebar.text_input("MongoDB URI (optional)", value="", help="Optional: mongodb://user:pass@host:port/db")

# HuggingFace model selection
HF_MODEL = st.sidebar.selectbox("Sentiment HF model", options=[
    "ProsusAI/finbert",  # try FinBERT — may need to be available
    "yiyanghkust/finbert-tone",  # alternative
    "cardiffnlp/twitter-roberta-base-sentiment"  # generic fallback
], index=0)

# ---------------------------
# Utility: Data fetchers
# ---------------------------
@st.cache_data(ttl=300)
def get_stock_data_yf(symbol: str, period="2y", interval="1d"):
    """Fetch historical OHLCV using yfinance."""
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        df = df.dropna()
        if df.empty:
            raise Exception("Empty dataframe from yfinance")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"yfinance fetch failed: {e}")
        # try Alpha Vantage fallback if key provided
        if ALPHA_VANTAGE_KEY:
            try:
                ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
                data, _ = ts.get_daily_adjusted(symbol, outputsize='full')
                data.columns = [c.replace(' ', '_') for c in data.columns]
                data = data.rename(columns={
                    '1. open':'Open','2. high':'High','3. low':'Low','4. close':'Close','5. adjusted close':'Adj Close',
                    '6. volume':'Volume'
                })
                data = data.sort_index()
                data.index = pd.to_datetime(data.index)
                return data
            except Exception as e2:
                st.error(f"Alpha Vantage fallback failed: {e2}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

# News fetcher (GNews)
@st.cache_data(ttl=300)
def get_news_gnews(query: str, max_results=20):
    """Simple news fetch using gnewsclient."""
    client = gnewsclient.NewsClient(language='english', location='global', max_results=max_results)
    try:
        client.query = query
        results = client.get_news()
        # results is list of dicts with keys: title, link, etc.
        # Normalize
        df = pd.DataFrame(results)
        return df
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return pd.DataFrame()

# ---------------------------
# Sentiment analysis
# ---------------------------
@st.cache_resource
def get_hf_pipeline(model_name):
    """Create a transformers sentiment pipeline. Cached to avoid reloading repeatedly."""
    try:
        # Use Auto tokenizer/model to allow FinBERT or alternative
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if MODEL_DEVICE=="cuda" else -1, return_all_scores=True)
        return pipe
    except Exception as e:
        st.warning(f"Could not load {model_name}: {e}. Falling back to simpler sentiment pipeline.")
        # fallback to a generic sentiment pipeline
        try:
            pipe = pipeline("sentiment-analysis", device=0 if MODEL_DEVICE=="cuda" else -1)
            return pipe
        except Exception as e2:
            st.error(f"Transformer fallback failed: {e2}")
            return None

def textblob_sentiment(text: str):
    t = TextBlob(text)
    return t.sentiment.polarity  # -1 .. 1

def hf_sentiment_score(pipe, text: str):
    """Return a numeric sentiment score normalized to -1..1. Works generically for many HF models."""
    if pipe is None:
        return 0.0
    try:
        out = pipe(text)
        # out could be e.g. [{'label':'POSITIVE','score':0.95}] or more complex if return_all_scores
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            # return_all_scores True -> list of dicts
            scores = out[0]
            # try to compute polarity: (pos - neg)
            pos = next((d['score'] for d in scores if 'POS' in d.get('label','').upper()), 0.0)
            neg = next((d['score'] for d in scores if 'NEG' in d.get('label','').upper()), 0.0)
            neu = next((d['score'] for d in scores if 'NEU' in d.get('label','').upper()), 0.0)
            val = pos - neg
            return float(val)
        else:
            # simple sentiment-analysis output: list of {'label','score'}
            entry = out[0]
            label = entry.get('label','').upper()
            score = float(entry.get('score',0.0))
            if 'NEG' in label:
                return -score
            elif 'POS' in label:
                return score
            elif 'NEU' in label:
                return 0.0
            else:
                # unknown label mapping (e.g., 1-star .. 5-star) -> try numeric mapping
                try:
                    # cardiffnlp labels may be 'LABEL_0' etc. We fallback to 0
                    return 0.0
                except:
                    return 0.0
    except Exception as e:
        st.warning(f"HF sentiment pipe failed: {e}")
        return 0.0

# ---------------------------
# Feature engineering
# ---------------------------
def add_technical_indicators(df: pd.DataFrame):
    df = df.copy()
    # Use the `ta` package to add RSI, SMA, MACD, EMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ROC'] = df['Close'].pct_change()
    # RSI
    try:
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    except Exception:
        df['RSI_14'] = df['Close'].rolling(14).apply(lambda x: 0)
    # MACD
    try:
        macd = ta.trend.macd_diff(df['Close'])
        df['MACD_diff'] = macd
    except Exception:
        df['MACD_diff'] = 0
    df = df.dropna()
    return df

# ---------------------------
# Dataset / Model (PyTorch LSTM)
# ---------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: [batch, seq_len, features]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last output
        out = self.fc(out)
        return out.squeeze(-1)

def create_sequences(df, feature_cols, target_col='Close', seq_len=20):
    arr = df[feature_cols].values
    targets = df[target_col].values
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(targets[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_lstm_model(X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32, lr=1e-3, device=MODEL_DEVICE):
    input_size = X_train.shape[2]
    model = LSTMModel(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_ds = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        if (epoch+1) % 5 == 0 or epoch==0:
            st.text(f"Epoch {epoch+1}/{epochs} - Train MSE: {epoch_loss:.6f}")
    return model

# ---------------------------
# Prediction & Decision logic
# ---------------------------
def predict_next_price(model, recent_seq, device=MODEL_DEVICE):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(recent_seq.astype(np.float32)).unsqueeze(0).to(device)  # [1, seq, features]
        pred = model(x).cpu().numpy().item()
        return float(pred)

def buy_sell_hold_logic(current_price, predicted_price, avg_sentiment, threshold_pct=0.005):
    """Simple rule:
      - If predicted_price > current_price * (1 + threshold) AND sentiment positive => BUY
      - If predicted_price < current_price * (1 - threshold) AND sentiment negative => SELL
      - Else HOLD
    """
    up = current_price * (1 + threshold_pct)
    down = current_price * (1 - threshold_pct)
    if predicted_price >= up and avg_sentiment > 0.05:
        return "BUY"
    elif predicted_price <= down and avg_sentiment < -0.05:
        return "SELL"
    else:
        return "HOLD"

# ---------------------------
# Backtesting with Backtrader
# ---------------------------
class SimpleStrategy(bt.Strategy):
    params = dict(predictions=None)
    def __init__(self):
        self.predictions = self.p.predictions or {}
    def next(self):
        # Use date string to check signal
        dt = self.data.datetime.date(0).isoformat()
        if dt in self.predictions:
            sig = self.predictions[dt]
            if sig == "BUY" and not self.position:
                self.buy()
            elif sig == "SELL" and self.position:
                self.sell()

def run_backtest(df, signals):
    """
    df: historical df with columns Open/High/Low/Close/Volume and Date index
    signals: dict mapping date (ISO) -> "BUY"/"SELL"/"HOLD"
    """
    cerebro = bt.Cerebro()
    # create data feed
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SimpleStrategy, predictions=signals)
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    start_val = cerebro.broker.getvalue()
    cerebro.run()
    end_val = cerebro.broker.getvalue()
    return start_val, end_val

# ---------------------------
# Optional MongoDB logging
# ---------------------------
def init_mongo(uri: str):
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # quick ping
        client.admin.command('ping')
        db = client.get_default_database() or client['nexttick']
        return db
    except Exception as e:
        st.warning(f"Could not connect to MongoDB: {e}")
        return None

def save_to_mongo(db, collection_name, doc):
    if db is None:
        return
    try:
        coll = db[collection_name]
        coll.insert_one(doc)
    except Exception as e:
        st.warning(f"Failed to save to MongoDB: {e}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("NextTick — Stock Prediction + Sentiment Dashboard")
st.markdown("Enter a stock symbol (e.g., AAPL) and click **Analyze**. This demo runs a small LSTM model on recent data and uses hybrid sentiment.")

col1, col2 = st.columns([2,1])
with col1:
    symbol = st.text_input("Stock symbol", value="AAPL").upper()
    lookback_period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    seq_len = st.slider("Sequence length (days) for LSTM", min_value=10, max_value=60, value=20)
    retrain_epochs = st.slider("Train epochs (quick demo)", min_value=5, max_value=50, value=15)
    analyze_btn = st.button("Analyze")
with col2:
    st.write("Model & Data")
    st.write(f"HF model: {HF_MODEL}")
    st.write(f"Device: {MODEL_DEVICE}")
    st.write("Options:")
    st.write("- Uses yfinance for historical data")
    st.write("- GNews for headlines")
    st.write("- HF + TextBlob for sentiment")
    st.write("- PyTorch LSTM for next-day price")

# Prepare HF pipeline (cached)
hf_pipe = get_hf_pipeline(HF_MODEL)

if analyze_btn and symbol:
    with st.spinner("Fetching data and news..."):
        df = get_stock_data_yf(symbol, period=lookback_period)
        if df.empty:
            st.error("No stock data found. Check symbol or try longer period.")
            st.stop()

        news_df = get_news_gnews(symbol, max_results=30)
        st.success(f"Fetched {len(df)} price rows and {len(news_df)} news items.")
    # Show price chart
    st.subheader(f"{symbol} Price Chart")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'], label='Close')
    ax.set_title(f"{symbol} Close Price")
    ax.legend()
    st.pyplot(fig)

    # Sentiment analysis over news headlines
    st.subheader("News & Sentiment")
    if not news_df.empty:
        news_df['text'] = news_df['title'].fillna('') + ". " + news_df.get('desc', '').fillna('')
        # compute sentiment
        tb_scores, hf_scores = [], []
        for t in news_df['title'].astype(str).tolist():
            tb_scores.append(textblob_sentiment(t))
            hf_scores.append(hf_sentiment_score(hf_pipe, t))
        news_df['textblob'] = tb_scores
        news_df['hf_sentiment'] = hf_scores
        # average
        news_df['avg_sentiment'] = news_df[['textblob','hf_sentiment']].mean(axis=1)
        avg_sentiment = float(news_df['avg_sentiment'].mean())
        st.write(f"Average sentiment (news): **{avg_sentiment:.4f}**")
        st.dataframe(news_df[['title','textblob','hf_sentiment','avg_sentiment']].head(10))
    else:
        avg_sentiment = 0.0
        st.info("No news found for this symbol via GNews.")

    # Feature engineering
    st.subheader("Feature engineering & training data")
    df_feat = add_technical_indicators(df)
    st.write(f"After indicators, {len(df_feat)} rows remain.")
    st.dataframe(df_feat.tail(5))

    # Prepare sequences
    feature_cols = ['Close','SMA_10','SMA_50','EMA_10','EMA_50','RSI_14','MACD_diff','ROC']
    df_feat = df_feat.dropna()
    if len(df_feat) < seq_len + 10:
        st.error("Not enough data after indicators for sequence modeling. Try longer historical period.")
        st.stop()

    X, y = create_sequences(df_feat, feature_cols, target_col='Close', seq_len=seq_len)
    # Split train/val
    split = int(len(X)*0.85)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    st.write(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Train model (quick)
    st.subheader("Training LSTM model")
    model = train_lstm_model(X_train, y_train, X_val, y_val, epochs=retrain_epochs)

    # Predict next-day price using last available sequence
    recent_seq = X[-1]  # last sequence
    predicted_price = predict_next_price(model, recent_seq)
    current_price = float(df_feat['Close'].iloc[-1])
    st.write(f"Current price: **{current_price:.4f}**  — Predicted next-day price: **{predicted_price:.4f}**")

    # Decision
    suggestion = buy_sell_hold_logic(current_price, predicted_price, avg_sentiment)
    st.subheader(f"Trading suggestion: **{suggestion}**")
    st.info(f"Rule: predicted vs current with sentiment. Threshold = 0.5%")

    # Backtesting (simple): map each date to "HOLD" except last day predicted signal
    st.subheader("Quick backtest (demo strategy)")
    # Create a simple signal series: for demo, create buy/sell on signal date
    signals = {}
    sig_date = df_feat.index[-1].date().isoformat()
    signals[sig_date] = suggestion
    start_val, end_val = run_backtest(df.reset_index().set_index('Date'), signals) if 'Date' in df.reset_index().columns else run_backtest(df, signals)
    st.write(f"Backtest start cash: ${start_val:.2f}, end value: ${end_val:.2f}")

    # Save results optionally to MongoDB
    db = init_mongo(MONGO_URI)
    log_doc = {
        "symbol": symbol,
        "timestamp": datetime.datetime.utcnow(),
        "current_price": current_price,
        "predicted_price": predicted_price,
        "suggestion": suggestion,
        "avg_sentiment": avg_sentiment,
        "news_count": len(news_df),
        "model": HF_MODEL
    }
    save_to_mongo(db, "predictions", log_doc)
    st.success("Analysis complete. (Results optionally saved to MongoDB)")

    # Show recommendation explanation
    st.subheader("Explainability & charts")
    # plot technical indicators window
    fig2, ax2 = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ax2[0].plot(df_feat.index[-120:], df_feat['Close'][-120:], label='Close')
    ax2[0].plot(df_feat.index[-120:], df_feat['SMA_10'][-120:], label='SMA_10')
    ax2[0].plot(df_feat.index[-120:], df_feat['SMA_50'][-120:], label='SMA_50')
    ax2[0].legend()
    ax2[1].plot(df_feat.index[-120:], df_feat['RSI_14'][-120:], label='RSI_14')
    ax2[1].axhline(70, linestyle='--'); ax2[1].axhline(30, linestyle='--')
    st.pyplot(fig2)

    # show sentiment time-series if we had many articles
    if not news_df.empty:
        st.subheader("News sentiment distribution")
        fig3, ax3 = plt.subplots(figsize=(8,3))
        ax3.hist(news_df['avg_sentiment'].dropna(), bins=15)
        ax3.set_xlabel("Avg sentiment")
        st.pyplot(fig3)

    st.balloons()
