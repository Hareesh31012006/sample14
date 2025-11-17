# -------------------------------------------------
# OPTIMIZED VERSION FOR DEPLOYMENT (Streamlit)
# -------------------------------------------------

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
import backtrader as bt
from pymongo import MongoClient
from alpha_vantage.timeseries import TimeSeries
import ta
import datetime
import os

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="NextTick - Optimized Stock Predictor",
)

MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.title("NextTick Settings")
st.sidebar.write(f"Running on: **{MODEL_DEVICE}**")

ALPHA_VANTAGE_KEY = st.sidebar.text_input("AlphaVantage Key (optional)", "")
MONGO_URI = st.sidebar.text_input("MongoDB URI (optional)", "")

HF_MODEL = st.sidebar.selectbox(
    "Sentiment model",
    ["ProsusAI/finbert", "yiyanghkust/finbert-tone", "cardiffnlp/twitter-roberta-base-sentiment"]
)

# -------------------------------------------------
# CACHED MODEL LOADING (MAJOR SPEED BOOST)
# -------------------------------------------------
@st.cache_resource
def load_hf_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if MODEL_DEVICE == "cuda" else -1
        )
        return pipe
    except:
        return pipeline("sentiment-analysis")

hf_pipe = load_hf_model(HF_MODEL)

# -------------------------------------------------
# STOCK DATA FETCH
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_stock_data(symbol, period="2y"):
    df = yf.download(symbol, period=period)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    return df

# -------------------------------------------------
# NEWS FETCH
# -------------------------------------------------
@st.cache_data(ttl=600)
def get_news(query):
    try:
        client = gnewsclient.NewsClient(language="english", location="global", max_results=20)
        client.query = query
        return pd.DataFrame(client.get_news())
    except:
        return pd.DataFrame()

# -------------------------------------------------
# SENTIMENT
# -------------------------------------------------
def get_sentiment(text):
    try:
        tb = TextBlob(text).sentiment.polarity
        hf = hf_pipe(text)[0]

        if "NEG" in hf["label"].upper():
            score = -hf["score"]
        elif "POS" in hf["label"].upper():
            score = hf["score"]
        else:
            score = 0

        return (tb + score) / 2
    except:
        return 0

# -------------------------------------------------
# TECHNICAL INDICATORS (CACHED)
# -------------------------------------------------
@st.cache_data
def add_indicators(df):
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["ROC"] = df["Close"].pct_change()
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD_diff"] = ta.trend.macd_diff(df["Close"])
    return df.dropna()

# -------------------------------------------------
# LSTM MODEL + DATASET
# -------------------------------------------------
class SeqData(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# -------------------------------------------------
# CREATE SEQUENCES
# -------------------------------------------------
def create_seq(df, feats, seq_len):
    X, y = [], []
    arr = df[feats].values
    target = df["Close"].values
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(target[i])
    return np.array(X), np.array(y)

# -------------------------------------------------
# TRAIN MODEL (CACHED FOR DEPLOYMENT SPEED)
# -------------------------------------------------
@st.cache_resource
def train_model_cached(symbol, seq_len, epochs, df_feat, feature_cols):
    X, y = create_seq(df_feat, feature_cols, seq_len)
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]

    model = LSTM(len(feature_cols)).to(MODEL_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(SeqData(X_train, y_train), batch_size=32, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(MODEL_DEVICE), yb.to(MODEL_DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model, X

# -------------------------------------------------
# BUY-SELL LOGIC
# -------------------------------------------------
def decision(cp, pp, sent):
    if pp > cp * 1.005 and sent > 0:
        return "BUY"
    if pp < cp * 0.995 and sent < 0:
        return "SELL"
    return "HOLD"

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🚀 NextTick – Optimized Stock Prediction Dashboard")

col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter stock symbol", "AAPL").upper()
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    seq_len = st.slider("Sequence length", 10, 60, 20)
    epochs = st.slider("Training epochs", 5, 40, 12)
    go = st.button("Analyze")

with col2:
    st.info("✔ LSTM Forecast\n✔ Hybrid Sentiment\n✔ Indicators\n✔ Backtest")

if go:
    st.subheader(f"📈 Fetching data for {symbol}...")
    df = get_stock_data(symbol, period)

    if df.empty:
        st.error("Symbol not found.")
        st.stop()

    news = get_news(symbol)

    # SENTIMENT
    if not news.empty:
        news["sentiment"] = news["title"].astype(str).apply(get_sentiment)
        avg_sent = news["sentiment"].mean()
    else:
        avg_sent = 0

    # INDICATORS
    df_feat = add_indicators(df)
    feats = ["Close","SMA_10","SMA_50","EMA_10","EMA_50","ROC","RSI_14","MACD_diff"]

    # MODEL TRAINING (CACHED)
    with st.spinner("Training LSTM model..."):
        model, X = train_model_cached(symbol, seq_len, epochs, df_feat, feats)

    # PREDICTION
    last_seq = X[-1]
    with torch.no_grad():
        pred = model(torch.tensor(last_seq).unsqueeze(0).to(MODEL_DEVICE)).cpu().numpy()[0]

    cp = df_feat["Close"].iloc[-1]
    sug = decision(cp, pred, avg_sent)

    st.success(f"Prediction: {pred:.2f}")
    st.success(f"Decision: **{sug}**")

    # PLOT
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_feat["Close"][-100:])
    st.pyplot(fig)
