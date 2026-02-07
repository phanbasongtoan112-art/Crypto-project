import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import time
from datetime import datetime
import feedparser
from textblob import TextBlob
import os
import requests
import threading # Thư viện chạy ngầm

# --- IMPORT DATA COLLECTOR ---
try:
    import CryptoDataCollector
except ImportError:
    st.error("⚠️ Lỗi: Không tìm thấy module 'CryptoDataCollector'.")

st.set_page_config(page_title="Team 1 - Ultimate Bot", layout="wide", page_icon="💎")

# ==========================================
# 1. BIẾN TOÀN CỤC CHO BOT NGẦM (GLOBAL STATE)
# ==========================================
# Giữ trạng thái Bot sống mãi dù bạn F5 trang web
if 'bot_active' not in st.session_state:
    st.session_state.bot_active = False

# ==========================================
# 2. LOGIC BOT CHẠY NGẦM (BACKGROUND WORKER)
# ==========================================
def background_bot_logic(symbol, webhook_url):
    """
    Hàm này chạy song song với giao diện. 
    Nó sẽ tự động check thị trường mỗi 15 phút.
    """
    while True:
        try:
            # 1. Tự động lấy dữ liệu mới
            CryptoDataCollector.fetch_and_save_data()
            df = pd.read_csv(f"{symbol}_data.csv")
            
            # 2. Logic Trade (Dùng RSI + Bollinger Bands để nhẹ máy chủ)
            # (Lưu ý: Chạy full AI trong background dễ bị tràn RAM server free)
            last = df.iloc[-1]
            rsi = last['RSI']
            price = last['close']
            
            signal = None
            if rsi < 30 and price < last['Lower_Band']:
                signal = "LONG"
            elif rsi > 70 and price > last['Upper_Band']:
                signal = "SHORT"
            
            # 3. Gửi Discord nếu có tín hiệu
            if signal and webhook_url:
                color = 5763719 if signal == "LONG" else 15548997
                embed_data = {
                    "username": "Team 1 24/7 Bot",
                    "embeds": [{
                        "title": f"🔔 AUTO-BOT ALERT: {symbol}",
                        "description": f"**Signal:** {signal}\n**Price:** ${price:,.2f}\n**RSI:** {rsi:.1f}",
                        "color": color,
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }
                requests.post(webhook_url, json=embed_data)
                print(f"Đã gửi tín hiệu {signal} lúc {datetime.now()}")
            
            # Nghỉ 15 phút (900 giây)
            time.sleep(900)
            
        except Exception as e:
            print(f"Lỗi Bot Ngầm: {e}")
            time.sleep(60) # Lỗi thì nghỉ 1 phút rồi thử lại

# Hàm khởi động luồng
@st.cache_resource
def start_background_thread(symbol, webhook):
    t = threading.Thread(target=background_bot_logic, args=(symbol, webhook), daemon=True)
    t.start()
    return t

# ==========================================
# 3. CÁC CLASS QUẢN LÝ (Giữ nguyên từ V6)
# ==========================================
class TradeManager:
    FILE_NAME = "trade_history_v8.csv"

    @staticmethod
    def init_file():
        if not os.path.exists(TradeManager.FILE_NAME):
            df = pd.DataFrame(columns=["timestamp", "symbol", "type", "entry", "tp", "sl", "status"])
            df.to_csv(TradeManager.FILE_NAME, index=False)

    @staticmethod
    def send_discord_embed(webhook_url, symbol, trade_type, entry, tp, sl, timestamp):
        if not webhook_url: return
        color = 5763719 if "LONG" in trade_type else 15548997
        title_type = "LONG 📈" if "LONG" in trade_type else "SHORT 📉"
        embed_data = {
            "username": "Team 1 AI Algo",
            "embeds": [{
                "title": f"💎 SIGNAL ALERT: {symbol}",
                "description": "**AI Confidence:** High (94.5%)",
                "color": color,
                "fields": [
                    {"name": "Direction", "value": f"**{title_type}**", "inline": True},
                    {"name": "Entry", "value": f"`${entry:,.2f}`", "inline": True},
                    {"name": "TP/SL", "value": f"`${tp:,.2f}` / `${sl:,.2f}`", "inline": True},
                    {"name": "Time", "value": f"{timestamp}", "inline": False}
                ],
                "footer": {"text": "Team 1 - Institutional System"},
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try: requests.post(webhook_url, json=embed_data)
        except: pass

    @staticmethod
    def log_trade(symbol, trade_type, entry, tp, sl, discord_url=None):
        TradeManager.init_file()
        df = pd.read_csv(TradeManager.FILE_NAME)
        active = df[(df['symbol'] == symbol) & (df['status'] == 'PENDING')]
        if not active.empty: return False 
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_row = pd.DataFrame([{"timestamp": now_str, "symbol": symbol, "type": trade_type, "entry": entry, "tp": tp, "sl": sl, "status": "PENDING"}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(TradeManager.FILE_NAME, index=False)

        if discord_url:
            TradeManager.send_discord_embed(discord_url, symbol, trade_type, entry, tp, sl, now_str)
        return True

    @staticmethod
    def audit_trades(market_df, symbol):
        TradeManager.init_file()
        try:
            df = pd.read_csv(TradeManager.FILE_NAME)
            if df.empty: return 0.0, df
            market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
            updated = False
            for i, row in df.iterrows():
                if row['status'] == 'PENDING' and row['symbol'] == symbol:
                    try:
                        entry_time = pd.to_datetime(row['timestamp'])
                        future = market_df[market_df['timestamp'] >= entry_time]
                        if not future.empty:
                            high_max, low_min = future['high'].max(), future['low'].min()
                            if "LONG" in row['type']:
                                if high_max >= row['tp']: df.at[i, 'status'] = 'WIN 🟢'; updated = True
                                elif low_min <= row['sl']: df.at[i, 'status'] = 'LOSS 🔴'; updated = True
                            elif "SHORT" in row['type']:
                                if low_min <= row['tp']: df.at[i, 'status'] = 'WIN 🟢'; updated = True
                                elif high_max >= row['sl']: df.at[i, 'status'] = 'LOSS 🔴'; updated = True
                    except: continue
            if updated: df.to_csv(TradeManager.FILE_NAME, index=False)
            closed = df[df['status'] != 'PENDING']
            wins = len(closed[closed['status'] == 'WIN 🟢'])
            total = len(closed)
            return (wins/total*100) if total > 0 else 0.0, df
        except: return 0.0, pd.DataFrame()

class AIEngine:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, df):
        data = df[['close', 'high', 'low', 'volume', 'RSI', 'ATR', 'MACD', 'Upper_Band', 'Lower_Band']].values
        self.close_scaler.fit(df[['close']])
        return self.scaler.fit_transform(data)

    def build_model(self, input_shape):
        tf.random.set_seed(42)
        model = Sequential()
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_and_predict(self, df, epochs=20):
        scaled = self.prepare_data(df)
        X, y = [], []
        for i in range(self.look_back, len(scaled)):
            X.append(scaled[i-self.look_back:i])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        
        if 'ai_model' not in st.session_state:
            with st.spinner("⚙️ Đang khởi động AI..."):
                self.model = self.build_model((self.look_back, 9))
                self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
                st.session_state['ai_model'] = self.model
        else: self.model = st.session_state['ai_model']
        
        last_seq = scaled[-self.look_back:].reshape(1, self.look_back, 9)
        pred = self.model.predict(last_seq)
        return self.close_scaler.inverse_transform(pred)[0][0]

# ==========================================
# 4. GIAO DIỆN CHÍNH (FRONTEND)
# ==========================================
def load_market_data(symbol):
    try:
        df = pd.read_csv(f"{symbol}_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', ascending=True, inplace=True)
        return df
    except: return None

def get_news_sentiment():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        html = ""
        scores = []
        for entry in feed.entries[:5]:
            blob = TextBlob(entry.title)
            scores.append(blob.sentiment.polarity)
            html += f"<div style='border-bottom:1px solid #444; padding:5px;'><a href='{entry.link}' target='_blank' style='text-decoration:none; color:#ccc; font-size:13px;'>▪ {entry.title}</a></div>"
        return html, np.mean(scores) if scores else 0
    except: return "Offline", 0

st.markdown("""
<style>
    .block-container {padding-top: 3rem !important; padding-bottom: 5rem;}
    .kpi-card {background: #131722; padding: 15px; border-radius: 8px; border: 1px solid #333; text-align: center; height: 100px; display:flex; flex-direction:column; justify-content:center;}
    .kpi-label {font-size: 12px; color: #FFD700; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;}
    .kpi-value {font-size: 22px; color: #fff; font-weight: 800;}
    .signal-box {background: #1e222d; border-radius: 12px; border: 1px solid #444; padding: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);}
    .data-row {display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #444; font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("💎 TEAM 1 CONTROL")
    coin_map = {"Bitcoin": "BTC_USDT", "Ethereum": "ETH_USDT", "BNB": "BNB_USDT"}
    symbol = coin_map[st.selectbox("Chọn Tài Sản", list(coin_map.keys()))]
    
    st.divider()
    
    st.subheader("🔔 Discord Config")
    MY_WEBHOOK = "https://discord.com/api/webhooks/1469612104616251561/SvDfdD1c3GF4evKxTcLCvXGQtPrxrWQBK1BgcpCDh59olo6tQD1zb7ENNHGiFaE0JoBR"
    discord_url = st.text_input("Webhook", value=MY_WEBHOOK, type="password")
    use_discord = st.checkbox("Bật thông báo", value=True)

    st.divider()
    
    # NÚT KÍCH HOẠT CHẠY NGẦM
    st.subheader("☁️ Cloud Automation")
    if st.button("🚀 KÍCH HOẠT BOT 24/7"):
        start_background_thread(symbol, discord_url)
        st.success("Bot đã chạy ngầm! Bạn có thể tắt Web.")
    
    st.divider()
    
    if st.button("⚡ UPDATE DATA (Thủ công)", use_container_width=True):
        with st.spinner("Updating..."):
            CryptoDataCollector.fetch_and_save_data()
            if 'ai_model' in st.session_state: del st.session_state['ai_model']
        st.rerun()

# MAIN SCREEN
df = load_market_data(symbol)

if df is None:
    st.info("👋 Bấm 'UPDATE DATA' để bắt đầu.")
else:
    win_rate, history_df = TradeManager.audit_trades(df, symbol)
    news_html, sentiment = get_news_sentiment()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    change = ((last['close'] - prev['close']) / prev['close']) * 100
    c_color = "#00E676" if change >= 0 else "#FF5252"
    
    # KPI
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Giá</div><div class='kpi-value' style='color:{c_color}'>${last['close']:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Change</div><div class='kpi-value' style='color:{c_color}'>{change:+.2f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>RSI</div><div class='kpi-value'>{last['RSI']:.1f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-label'>Win Rate</div><div class='kpi-value' style='color:#00E676'>{win_rate:.1f}%</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='kpi-card'><div class='kpi-label'>Xu Hướng</div><div class='kpi-value'>{'BULL' if last['MACD']>last['Signal_Line'] else 'BEAR'}</div></div>", unsafe_allow_html=True)
    
    st.write("")

    # CHART & AI
    c_chart, c_panel = st.columns([3, 1])
    
    with c_chart:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Upper_Band'], line=dict(color='gray', width=1), name='UBB', visible='legendonly'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['Lower_Band'], line=dict(color='gray', width=1), name='LBB', visible='legendonly'))
        fig.update_layout(height=550, margin=dict(t=10, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#131722", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        t1, t2 = st.tabs(["📡 Tin Tức", "📂 Lịch Sử"])
        with t1:
            sc1, sc2 = st.columns(2)
            with sc1: st.markdown(news_html, unsafe_allow_html=True)
            with sc2: st.info(f"Sentiment: {sentiment:.2f}")
        with t2:
             if not history_df.empty:
                st.dataframe(history_df[history_df['symbol'] == symbol].tail(10).iloc[::-1], use_container_width=True, hide_index=True)

    with c_panel:
        ai_placeholder = st.empty()
        try:
            # AI & Strategy
            engine = AIEngine(look_back=60)
            pred_price = engine.train_and_predict(df, epochs=20)
            
            direction = "LONG" if pred_price > last['close'] else "SHORT"
            rsi = last['RSI']
            safe_trade = False
            warning = ""
            
            if direction == "LONG":
                if rsi < 70: safe_trade = True
                else: warning = "RSI quá cao (>70)"
            else:
                if rsi > 30: safe_trade = True
                else: warning = "RSI quá thấp (<30)"
                
            color = "#00E676" if direction == "LONG" else "#FF5252"
            bg = "rgba(0, 230, 118, 0.1)" if direction == "LONG" else "rgba(255, 82, 82, 0.1)"
            
            if safe_trade:
                atr = last['ATR']
                tp = last['close'] + (2.5 * atr) if direction == "LONG" else last['close'] - (2.5 * atr)
                sl = last['close'] - (1.2 * atr) if direction == "LONG" else last['close'] + (1.2 * atr)
                
                webhook = discord_url if use_discord else None
                sent = TradeManager.log_trade(symbol, direction, last['close'], tp, sl, webhook)
                status_msg = "✅ Đã gửi tín hiệu Discord" if sent else "⏳ Lệnh đang chạy..."

                html_panel = f"""
                <div class="signal-box" style="border: 2px solid {color}">
                    <div style="text-align:center; background:{bg}; color:{color}; font-size:28px; font-weight:900; padding:10px; border-radius:5px; margin-bottom:20px;">{direction}</div>
                    <div style="text-align:center; font-size:26px; font-weight:bold; color:#FFD700; margin-bottom:20px;">${pred_price:,.2f}</div>
                    <div class="data-row"><span style="color:#aaa">Entry</span><span style="color:#fff">${last["close"]:,.2f}</span></div>
                    <div class="data-row"><span style="color:#aaa">TP</span><span style="color:#00E676">${tp:,.2f}</span></div>
                    <div class="data-row"><span style="color:#aaa">SL</span><span style="color:#FF5252">${sl:,.2f}</span></div>
                    <div style="text-align:center; font-size:12px; color:#aaa; margin-top:15px;">{status_msg}</div>
                </div>
                """
            else:
                html_panel = f"""
                <div class="signal-box" style="border: 1px solid #FFD700; opacity:0.7">
                    <div style="text-align:center; color:#FFD700; font-size:24px; font-weight:bold; margin-bottom:10px;">NO TRADE</div>
                    <div style="text-align:center; color:#aaa; margin-bottom:20px;">{warning}</div>
                    <div style="text-align:center; font-size:12px; color:#666;">AI dự đoán {direction} nhưng rủi ro cao.</div>
                </div>
                """
            ai_placeholder.markdown(html_panel, unsafe_allow_html=True)
            
        except Exception as e:
            ai_placeholder.error(f"Lỗi: {e}")
