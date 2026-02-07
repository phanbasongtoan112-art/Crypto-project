import streamlit as st
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Stable Cast", layout="wide", page_icon="💎")

# --- 2. CSS FRONTEND ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important; background-color: transparent !important;}
    .stApp {background-color: #0E1117;}
    .login-container {position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 400px; padding: 40px; background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 16px; text-align: center;}
    .avatar-frame {display: flex; justify-content: center; margin-bottom: 15px;}
    .avatar-img {width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #58a6ff; box-shadow: 0 0 15px rgba(88, 166, 255, 0.5);}
    .kpi-card {background: #161b22; padding: 15px; border-radius: 6px; border: 1px solid #30363d; text-align: center;}
    .kpi-label {color: #8b949e; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px; margin-bottom: 5px;}
    .kpi-value {color: #f0f6fc; font-size: 24px; font-weight: 700;}
    .stButton > button {border: 1px solid #30363d; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION & IMPORTS ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_info' not in st.session_state: st.session_state['user_info'] = None

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from datetime import datetime
import feedparser
from textblob import TextBlob
import os
import requests
import threading
import sqlite3
import hashlib
import base64
from io import BytesIO
from PIL import Image

try: import CryptoDataCollector
except ImportError: pass

# ==========================================
# 4. DATABASE & PROFILE HANDLER (MỚI)
# ==========================================
DB_FILE = "stable_cast.db"

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT, status TEXT)''')
    try: c.execute("ALTER TABLE users ADD COLUMN avatar TEXT"); 
    except: pass
    try: c.execute("INSERT INTO users (username, password, role, status) VALUES (?, ?, ?, ?)", ("admin", hashlib.sha256("admin123".encode()).hexdigest(), "admin", "active")); conn.commit()
    except: pass
    conn.close()

init_db()

# --- HÀM XỬ LÝ ẢNH ---
def image_to_base64(image):
    buffered = BytesIO(); image.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode()

def update_avatar(username, image_file):
    try:
        img = Image.open(image_file).resize((150, 150)); img_str = image_to_base64(img)
        conn = sqlite3.connect(DB_FILE); conn.execute("UPDATE users SET avatar=? WHERE username=?", (img_str, username)); conn.commit(); conn.close(); return True
    except: return False

def get_user_avatar(username):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor(); c.execute("SELECT avatar FROM users WHERE username=?", (username,)); data = c.fetchone(); conn.close(); return data[0] if data else None

# --- HÀM XỬ LÝ USER (LOGIN, UPDATE PASS, UPDATE NAME) ---
def check_login(u, p):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor(); c.execute("SELECT password, role, status FROM users WHERE username=?", (u,)); data = c.fetchone(); conn.close()
    if data:
        if data[2] == "banned": return "banned", None
        if data[0] == hashlib.sha256(p.encode()).hexdigest(): return True, data[1]
    return False, None

def create_user(u, p):
    conn = sqlite3.connect(DB_FILE)
    try: conn.execute("INSERT INTO users (username, password, role, status) VALUES (?, ?, ?, ?)", (u, hashlib.sha256(p.encode()).hexdigest(), "user", "active")); conn.commit(); return True
    except: return False
    finally: conn.close()

def change_password(username, old_pass, new_pass):
    success, _ = check_login(username, old_pass)
    if not success: return False # Sai mật khẩu cũ
    try:
        new_hash = hashlib.sha256(new_pass.encode()).hexdigest()
        conn = sqlite3.connect(DB_FILE); conn.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username)); conn.commit(); conn.close()
        return True
    except: return False

def change_username(current_username, new_username):
    try:
        conn = sqlite3.connect(DB_FILE)
        # Check trùng tên
        cursor = conn.cursor(); cursor.execute("SELECT * FROM users WHERE username=?", (new_username,))
        if cursor.fetchone(): conn.close(); return False
        
        # Đổi tên
        conn.execute("UPDATE users SET username=? WHERE username=?", (new_username, current_username))
        conn.commit(); conn.close()
        return True
    except: return False

def get_all_users():
    conn = sqlite3.connect(DB_FILE); df = pd.read_sql("SELECT username, role, status FROM users", conn); conn.close(); return df

def update_user_status(u, act):
    conn = sqlite3.connect(DB_FILE)
    if act == "ban": conn.execute("UPDATE users SET status='banned' WHERE username=?", (u,))
    elif act == "unban": conn.execute("UPDATE users SET status='active' WHERE username=?", (u,))
    elif act == "delete": conn.execute("DELETE FROM users WHERE username=?", (u,))
    conn.commit(); conn.close()

# ==========================================
# 5. LOGIC CORE
# ==========================================
class TradeManager:
    FILE_NAME = "trade_history_v26.csv"
    @staticmethod
    def init_file():
        if not os.path.exists(TradeManager.FILE_NAME): pd.DataFrame(columns=["timestamp", "symbol", "type", "entry", "tp", "sl", "status", "confidence", "user"]).to_csv(TradeManager.FILE_NAME, index=False)
    @staticmethod
    def reset_history():
        if os.path.exists(TradeManager.FILE_NAME): os.remove(TradeManager.FILE_NAME); TradeManager.init_file()
    @staticmethod
    def log_trade(symbol, trade_type, entry, tp, sl, conf, user, discord_url=None):
        TradeManager.init_file(); df = pd.read_csv(TradeManager.FILE_NAME)
        active = df[(df['symbol'] == symbol) & (df['status'] == 'PENDING') & (df['user'] == user)]
        if not active.empty: return False
        
        new_row = pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": trade_type, "entry": entry, "tp": tp, "sl": sl, "status": "PENDING", "confidence": conf, "user": user}])
        df = pd.concat([df, new_row], ignore_index=True); df.to_csv(TradeManager.FILE_NAME, index=False)

        if discord_url:
            color = 3447003 if "LONG" in trade_type else 15158332
            requests.post(discord_url, json={"embeds": [{"title": f"💎 SIGNAL ALERT: {symbol}", "description": f"**Trader:** {user}\n**Conf:** {conf:.1f}%", "color": color, "fields": [{"name": "Direction", "value": f"**{trade_type}**", "inline": True}, {"name": "Entry", "value": f"`${entry:,.2f}`", "inline": True}, {"name": "TP / SL", "value": f"`${tp:,.2f}` / `${sl:,.2f}`", "inline": True}]}]})
        return True
    @staticmethod
    def audit_trades(market_df, symbol):
        TradeManager.init_file()
        try:
            df = pd.read_csv(TradeManager.FILE_NAME); df['timestamp'] = pd.to_datetime(df['timestamp']); updated = False
            if df.empty: return 0.0, df
            return 0.0, df 
        except: return 0.0, pd.DataFrame()

class AIEngine:
    def __init__(self, look_back=60):
        self.look_back = look_back; self.scaler = MinMaxScaler((0,1)); self.close_scaler = MinMaxScaler((0,1)); self.model = None
    def prepare_data(self, df):
        data = df[['close', 'high', 'low', 'volume', 'RSI', 'ATR', 'MACD', 'Upper_Band', 'Lower_Band']].values
        self.close_scaler.fit(df[['close']]); return self.scaler.fit_transform(data)
    def build_model(self, input_shape):
        tf.random.set_seed(42); m = Sequential(); m.add(LSTM(64, return_sequences=False, input_shape=input_shape)); m.add(Dense(32, activation='relu')); m.add(Dense(1)); m.compile(optimizer='adam', loss='mse'); return m
    def train_and_predict(self, df, epochs=20):
        scaled = self.prepare_data(df); X, y = [], []
        for i in range(self.look_back, len(scaled)): X.append(scaled[i-self.look_back:i]); y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        if 'ai_model' not in st.session_state:
            self.model = self.build_model((self.look_back, 9)); self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False); st.session_state['ai_model'] = self.model
        else: self.model = st.session_state['ai_model']
        last_seq = scaled[-self.look_back:].reshape(1, self.look_back, 9); pred = self.model.predict(last_seq)
        return self.close_scaler.inverse_transform(pred)[0][0]

def calculate_confidence(df, direction):
    last = df.iloc[-1]; score = 50.0; rsi = last['RSI']
    if direction == "LONG": score += (30 - rsi)*1.5 if rsi < 30 else 0
    elif direction == "SHORT": score += (rsi - 70)*1.5 if rsi > 70 else 0
    return max(15.0, min(98.5, score))

def background_bot_logic(symbol, webhook_url):
    while True:
        try:
            CryptoDataCollector.fetch_and_save_data(); df = pd.read_csv(f"{symbol}_data.csv"); last = df.iloc[-1]; signal = None
            if last['RSI'] < 30: signal = "LONG"
            elif last['RSI'] > 70: signal = "SHORT"
            if signal and webhook_url:
                conf = calculate_confidence(df, signal)
                if not os.path.exists("trade_history_v26.csv"): df_hist = pd.DataFrame(columns=["status"])
                else: df_hist = pd.read_csv("trade_history_v26.csv")
                active = df_hist[(df_hist['symbol'] == symbol) & (df_hist['status'] == 'PENDING')]
                if active.empty:
                    atr = last['ATR']; tp = last['close'] + (2.5 * atr) if signal == "LONG" else last['close'] - (2.5 * atr); sl = last['close'] - (1.2 * atr) if signal == "LONG" else last['close'] + (1.2 * atr); color = 3447003 if signal == "LONG" else 15158332
                    requests.post(webhook_url, json={"embeds": [{"title": f"🔔 URGENT BOT: {symbol}", "description": f"**Signal:** {signal}\n**Confidence:** {conf:.1f}%", "color": color, "fields": [{"name": "Entry", "value": f"${last['close']:,.2f}", "inline": True}, {"name": "TP / SL", "value": f"${tp:,.2f} / ${sl:,.2f}", "inline": True}]}]})
            time.sleep(900)
        except: time.sleep(60)

@st.cache_resource
def start_background_thread(symbol, webhook):
    t = threading.Thread(target=background_bot_logic, args=(symbol, webhook), daemon=True); t.start()

def get_news_sentiment():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss"); html = ""; scores = []
        for entry in feed.entries[:5]: html += f"<div style='border-bottom:1px solid #444; padding:5px;'>▪ {entry.title}</div>"
        return html, 0
    except: return "Offline", 0

# ==========================================
# 6. UI RENDERER
# ==========================================
main_container = st.empty()
SYSTEM_WEBHOOK = "https://discord.com/api/webhooks/1469612104616251561/SvDfdD1c3GF4evKxTcLCvXGQtPrxrWQBK1BgcpCDh59olo6tQD1zb7ENNHGiFaE0JoBR"

def render_login():
    with main_container.container():
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<div style='text-align:center; padding-top:50px;'><h1 style='color:#fff;'>STABLE CAST</h1></div>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["LOGIN", "REGISTER"])
            with tab1:
                u = st.text_input("Username", key="l_u"); p = st.text_input("Password", type="password", key="l_p")
                if st.button("LOGIN"):
                    success, role = check_login(u, p)
                    if success == "banned": st.error("BANNED")
                    elif success: st.session_state['logged_in'] = True; st.session_state['user_info'] = (u, role); st.rerun()
                    else: st.error("INVALID")
            with tab2:
                nu = st.text_input("New User", key="r_u"); np = st.text_input("New Pass", type="password", key="r_p")
                if st.button("CREATE"):
                    if create_user(nu, np): st.success("SUCCESS"); time.sleep(1)
                    else: st.error("EXISTS")

def render_dashboard():
    main_container.empty()
    user, role = st.session_state['user_info']
    avatar_b64 = get_user_avatar(user)
    img_src = f"data:image/png;base64,{avatar_b64}" if avatar_b64 else "https://cdn-icons-png.flaticon.com/512/847/847969.png"
    
    with st.sidebar:
        st.markdown(f"<div class='avatar-frame'><img src='{img_src}' class='avatar-img'></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{user}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#888; font-size:12px;'>Role: {role.upper()}</p>", unsafe_allow_html=True)
        
        # --- MENU QUẢN LÝ HỒ SƠ ---
        with st.expander("👤 Manage Profile"):
            tab_ava, tab_name, tab_sec = st.tabs(["Avatar", "Name", "Security"])
            
            with tab_ava:
                st.caption("Change Profile Picture")
                uploaded = st.file_uploader("Upload", type=['png','jpg'], label_visibility="collapsed")
                if uploaded and update_avatar(user, uploaded): st.success("Updated!"); time.sleep(1); st.rerun()
            
            with tab_name:
                st.caption("Change Username")
                new_u = st.text_input("New Name", label_visibility="collapsed")
                if st.button("Rename"):
                    if change_username(user, new_u):
                        st.success("Success! Please Login again.")
                        time.sleep(1)
                        st.session_state['logged_in'] = False
                        st.rerun()
                    else: st.error("Name Taken")

            with tab_sec:
                st.caption("Change Password")
                old_p = st.text_input("Old Pass", type="password")
                new_p = st.text_input("New Pass", type="password")
                if st.button("Update Pass"):
                    if change_password(user, old_p, new_p): st.success("Success!")
                    else: st.error("Wrong Old Pass")

        st.divider()
        if st.button("Sign Out"): st.session_state['logged_in'] = False; st.rerun()
        if role == 'admin' and st.checkbox("🛡️ Admin"):
            st.dataframe(get_all_users(), hide_index=True)
            t = st.selectbox("User", get_all_users()['username'])
            a = st.selectbox("Act", ["ban", "unban", "delete"])
            if st.button("Do"): update_user_status(t, a); st.rerun()
            return

        st.header("💎 ASSETS")
        symbol = st.selectbox("Select", ["BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "PAXG_USDT"])
        
        wb = SYSTEM_WEBHOOK 
        if role == 'admin':
            wb = st.text_input("Webhook", value=SYSTEM_WEBHOOK, type="password")
            if st.button("Test"): requests.post(wb, json={"content":"Test OK"})
            if st.button("Reset"): TradeManager.reset_history()
            if st.button("Bot 24/7"): start_background_thread(symbol, wb)
        
        if st.button("Update Data"):
            with st.spinner("Sync..."): CryptoDataCollector.fetch_and_save_data(); st.session_state.pop('ai_model', None); st.rerun()

    try: df = pd.read_csv(f"{symbol}_data.csv"); df['timestamp'] = pd.to_datetime(df['timestamp']); df.sort_values('timestamp', inplace=True)
    except: st.info("Need Update."); return

    win, history = TradeManager.audit_trades(df, symbol)
    last = df.iloc[-1]; color = "#00E676"
    
    # KPI + Labels
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>PRICE</div><div class='kpi-value' style='color:{color}'>${last['close']:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>CHANGE</div><div class='kpi-value' style='color:{color}'>+0.5%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>RSI</div><div class='kpi-value'>{last['RSI']:.1f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-label'>WIN RATE</div><div class='kpi-value' style='color:#00E676'>{win:.1f}%</div></div>", unsafe_allow_html=True)

    c_chart, c_panel = st.columns([3, 1])
    with c_chart:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])])
        fig.update_layout(height=500, margin=dict(t=20, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history.tail(5) if not history.empty else history, use_container_width=True)

    with c_panel:
        ai = st.empty()
        try:
            eng = AIEngine(60); pred = eng.train_and_predict(df, 20)
            direct = "LONG" if pred > last['close'] else "SHORT"
            conf = calculate_confidence(df, direct)
            safe = (direct=="LONG" and last['RSI']<70) or (direct=="SHORT" and last['RSI']>30)
            col_sig = "#00E676" if direct == "LONG" else "#FF5252"
            
            if safe:
                atr = last['ATR']; tp = last['close']+(2.5*atr); sl = last['close']-(1.2*atr)
                conf_color = "#00E676" if conf > 80 else "#FFD700" if conf > 50 else "#FF5252"
                
                html = f"""
<div style="background:#0d1117; border-radius:8px; border:1px solid #30363d; padding:20px; border-top: 3px solid {col_sig}">
<div style="font-size:32px; font-weight:800; color:{col_sig}; text-align:center;">{direct}</div>
<div style="text-align:center; color:#888; font-size:12px;">Target: ${pred:,.2f}</div>
<div style="margin:15px 0;">
<div style="display:flex; justify-content:space-between; font-size:11px; color:#aaa;"><span>CONFIDENCE</span><span style="color:{conf_color}">{conf:.1f}%</span></div>
<div style="background:#333; height:4px; border-radius:2px;"><div style="width:{conf}%; background:{conf_color}; height:100%;"></div></div>
</div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">ENTRY</span><span style="color:#fff">${last['close']:,.2f}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">TP</span><span style="color:#00E676">${tp:,.2f}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">SL</span><span style="color:#FF5252">${sl:,.2f}</span></div>
</div>
"""
                ai.markdown(html, unsafe_allow_html=True)
                
                if st.button("🚀 PUSH ALERT TO DISCORD", use_container_width=True):
                    if TradeManager.log_trade(symbol, direct, last['close'], tp, sl, conf, user, wb):
                        st.toast(f"✅ Alert Sent by {user}!", icon="🚀")
                    else:
                        st.toast("⚠️ You already sent this signal!", icon="🚫")

            else:
                ai.markdown(f"""<div style="background:#0d1117; padding:20px; text-align:center; border-radius:8px; opacity:0.6"><h3>NO TRADE</h3><p>Risk High</p></div>""", unsafe_allow_html=True)
        except Exception as e: ai.error(f"Error: {e}")

if st.session_state['logged_in']: render_dashboard()
else: render_login()
