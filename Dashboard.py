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
import threading
import random

# --- IMPORT DATA COLLECTOR ---
try:
    import CryptoDataCollector
except ImportError:
    st.error("⚠️ Lỗi: Không tìm thấy module 'CryptoDataCollector'.")

st.set_page_config(page_title="Team 1 - Pro Terminal", layout="wide", page_icon="💎")

# ==========================================
# 1. HÀM TÍNH ĐIỂM (SMART SCORING)
# ==========================================
def calculate_confidence(df, direction):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50.0 
    
    # 1. RSI Dynamic
    rsi = last['RSI']
    if direction == "LONG":
        if rsi < 30: score += (30 - rsi) * 1.5 
        elif rsi > 50: score -= (rsi - 50) * 0.5
    elif direction == "SHORT":
        if rsi > 70: score += (rsi - 70) * 1.5
        elif rsi < 50: score -= (50 - rsi) * 0.5

    # 2. MACD
    macd_hist = last['MACD'] - last['Signal_Line']
    if direction == "LONG":
        if macd_hist > 0: score += 15 
        elif macd_hist > prev['MACD'] - prev['Signal_Line']: score += 5
        else: score -= 10
    elif direction == "SHORT":
        if macd_hist < 0: score += 15
        elif macd_hist < prev['MACD'] - prev['Signal_Line']: score += 5 
        else: score -= 10

    # 3. Bollinger Bands
    if direction == "LONG" and last['close'] <= last['Lower_Band']: score += 10
    elif direction == "SHORT" and last['close'] >= last['Upper_Band']: score += 10
    
    # 4. Volume
    vol_ma = df['volume'].tail(20).mean()
    if last['volume'] > vol_ma * 1.5: score += 5

    return max(15.0, min(98.5, score))

# ==========================================
# 2. BOT NGẦM (BACKGROUND)
# ==========================================
def background_bot_logic(symbol, webhook_url):
    while True:
        try:
            CryptoDataCollector.fetch_and_save_data()
            df = pd.read_csv(f"{symbol}_data.csv")
            last = df.iloc[-1]
            
            # Logic Bot Ngầm (An toàn)
            signal = None
            if last['RSI'] < 30: signal = "LONG"
            elif last['RSI'] > 70: signal = "SHORT"
            
            # Chỉ gửi nếu có tín hiệu mạnh
            if signal and webhook_url:
                conf = calculate_confidence(df, signal)
                color = 5763719 if signal == "LONG" else 15548997
                
                # Check file để tránh spam
                if not os.path.exists("trade_history_v10.csv"):
                    df_hist = pd.DataFrame(columns=["status"])
                else:
                    df_hist = pd.read_csv("trade_history_v10.csv")
                
                # Nếu chưa có lệnh Pending thì mới bắn
                active = df_hist[(df_hist['symbol'] == symbol) & (df_hist['status'] == 'PENDING')]
                
                if active.empty:
                    requests.post(webhook_url, json={
                        "embeds": [{
                            "title": f"🔔 AUTO-BOT ALERT: {symbol}",
                            "description": f"**Signal:** {signal}\n**Price:** ${last['close']}\n**Confidence:** {conf:.1f}%",
                            "color": color,
                            "footer": {"text": "Background Service"}
                        }]
                    })
            
            time.sleep(900) # Nghỉ 15 phút
        except: time.sleep(60)

@st.cache_resource
def start_background_thread(symbol, webhook):
    t = threading.Thread(target=background_bot_logic, args=(symbol, webhook), daemon=True)
    t.start()

# ==========================================
# 3. TRADE MANAGER
# ==========================================
class TradeManager:
    FILE_NAME = "trade_history_v10.csv"

    @staticmethod
    def init_file():
        if not os.path.exists(TradeManager.FILE_NAME):
            df = pd.DataFrame(columns=["timestamp", "symbol", "type", "entry", "tp", "sl", "status", "confidence"])
            df.to_csv(TradeManager.FILE_NAME, index=False)
            
    @staticmethod
    def reset_history():
        if os.path.exists(TradeManager.FILE_NAME):
            os.remove(TradeManager.FILE_NAME)
            TradeManager.init_file()
            return True
        return False

    @staticmethod
    def send_discord_embed(webhook_url, symbol, trade_type, entry, tp, sl, timestamp, conf):
        if not webhook_url: return
        color = 5763719 if "LONG" in trade_type else 15548997
        title_type = "LONG 📈" if "LONG" in trade_type else "SHORT 📉"
        quality = "Rất cao 🔥" if conf > 80 else "Trung bình ⚠️" if conf < 50 else "Ổn định ✅"
        
        embed_data = {
            "username": "Team 1 AI Algo",
            "embeds": [{
                "title": f"💎 SIGNAL ALERT: {symbol}",
                "description": f"**AI Confidence:** {conf:.1f}% ({quality})",
                "color": color,
                "fields": [
                    {"name": "Direction", "value": f"**{title_type}**", "inline": True},
                    {"name": "Entry", "value": f"`${entry:,.2f}`", "inline": True},
                    {"name": "TP / SL", "value": f"`${tp:,.2f}` / `${sl:,.2f}`", "inline": True},
                    {"name": "Time", "value": f"{timestamp}", "inline": False}
                ],
                "footer": {"text": "Team 1 - Institutional System"},
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try: requests.post(webhook_url, json=embed_data)
        except: pass

    @staticmethod
    def log_trade(symbol, trade_type, entry, tp, sl, conf, discord_url=None):
        TradeManager.init_file()
        df = pd.read_csv(TradeManager.FILE_NAME)
        
        # --- QUAN TRỌNG: CHỐNG SPAM ---
        # Nếu đã có lệnh PENDING của coin này -> KHÔNG GỬI NỮA
        active = df[(df['symbol'] == symbol) & (df['status'] == 'PENDING')]
        if not active.empty: 
            return False # Trả về False để báo là không gửi
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_row = pd.DataFrame([{
            "timestamp": now_str, "symbol": symbol, "type": trade_type, 
            "entry": entry, "tp": tp, "sl": sl, "status": "PENDING", "confidence": conf
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(TradeManager.FILE_NAME, index=False)

        if discord_url:
            TradeManager.send_discord_embed(discord_url, symbol, trade_type, entry, tp, sl, now_str, conf)
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

# ==========================================
# 4. AI ENGINE
# ==========================================
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
            with st.spinner("⚙️ AI đang phân tích dữ liệu..."):
                self.model = self.build_model((self.look_back, 9))
                self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False)
                st.session_state['ai_model'] = self.model
        else: self.model = st.session_state['ai_model']
        
        last_seq = scaled[-self.look_back:].reshape(1, self.look_back, 9)
        pred = self.model.predict(last_seq)
        return self.close_scaler.inverse_transform(pred)[0][0]

# ==========================================
# 5. GIAO DIỆN CHÍNH
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

with st.sidebar:
    st.header("💎 TEAM 1 CONTROL")
    
    # 1. DANH SÁCH COIN ĐẦY ĐỦ (Đã thêm lại)
    coin_map = {
        "Bitcoin": "BTC_USDT", 
        "Ethereum": "ETH_USDT", 
        "BNB": "BNB_USDT",
        "Solana": "SOL_USDT",
        "Gold (PAXG)": "PAXG_USDT"
    }
    symbol = coin_map[st.selectbox("Chọn Tài Sản", list(coin_map.keys()))]
    
    st.divider()
    st.subheader("🔔 Discord")
    MY_WEBHOOK = "https://discord.com/api/webhooks/1469612104616251561/SvDfdD1c3GF4evKxTcLCvXGQtPrxrWQBK1BgcpCDh59olo6tQD1zb7ENNHGiFaE0JoBR"
    discord_url = st.text_input("Webhook", value=MY_WEBHOOK, type="password")
    use_discord = st.checkbox("Bật thông báo", value=True)
    
    # Nút Test Discord (Mới)
    if st.button("🔔 TEST KẾT NỐI"):
        try:
            requests.post(discord_url, json={"content": "✅ **Test connection from Team 1 Bot successful!**"})
            st.success("Đã gửi tin nhắn test!")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    # Nút Xóa Lịch Sử (Mới)
    if st.button("🗑️ RESET LỊCH SỬ"):
        TradeManager.reset_history()
        st.success("Đã xóa lệnh cũ! Bot sẽ bắn lệnh mới ngay.")
        time.sleep(1)
        st.rerun()

    st.divider()
    if st.button("🚀 KÍCH HOẠT BOT 24/7"):
        start_background_thread(symbol, discord_url)
        st.success("Bot đã chạy ngầm!")
    
    st.divider()
    if st.button("⚡ UPDATE DATA", use_container_width=True):
        with st.spinner("Updating..."):
            CryptoDataCollector.fetch_and_save_data()
            if 'ai_model' in st.session_state: del st.session_state['ai_model']
        st.success("Xong!")
        time.sleep(0.5)
        st.rerun()

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
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Giá</div><div class='kpi-value' style='color:{c_color}'>${last['close']:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Change</div><div class='kpi-value' style='color:{c_color}'>{change:+.2f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>RSI</div><div class='kpi-value'>{last['RSI']:.1f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-label'>Win Rate</div><div class='kpi-value' style='color:#00E676'>{win_rate:.1f}%</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='kpi-card'><div class='kpi-label'>Xu Hướng</div><div class='kpi-value'>{'BULL' if last['MACD']>last['Signal_Line'] else 'BEAR'}</div></div>", unsafe_allow_html=True)
    
    st.write("")

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
            engine = AIEngine(look_back=60)
            pred_price = engine.train_and_predict(df, epochs=20)
            
            direction = "LONG" if pred_price > last['close'] else "SHORT"
            confidence = calculate_confidence(df, direction)
            
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
                sent = TradeManager.log_trade(symbol, direction, last['close'], tp, sl, confidence, webhook)
                
                if sent:
                    status_msg = "✅ Đã gửi tín hiệu Discord"
                else:
                    # NẾU KHÔNG GỬI -> CÓ THỂ DO LỆNH TRÙNG
                    status_msg = "⛔ Lệnh cũ đang chạy (Spam filter)"
                
                conf_color = "#00E676" if confidence > 80 else "#FFD700" if confidence > 50 else "#FF5252"

                html_panel = f"""
<div class="signal-box" style="border: 2px solid {color}">
<div style="text-align:center; background:{bg}; color:{color}; font-size:28px; font-weight:900; padding:10px; border-radius:5px; margin-bottom:10px;">{direction}</div>
<div style="text-align:center; font-size:26px; font-weight:bold; color:#FFD700; margin-bottom:15px;">${pred_price:,.2f}</div>
<div style="margin-bottom:15px;">
<div style="display:flex; justify-content:space-between; font-size:12px; color:#aaa;">
<span>AI Confidence</span>
<span style="color:{conf_color}">{confidence:.1f}%</span>
</div>
<div style="background:#333; height:8px; border-radius:4px; overflow:hidden;">
<div style="background:{conf_color}; width:{confidence}%; height:100%;"></div>
</div>
</div>
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
