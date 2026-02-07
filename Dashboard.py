import streamlit as st
import time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Stable Cast Ecosystem", layout="wide", page_icon="💎")

# --- 2. CSS FRONTEND ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important; background-color: transparent !important;}
    .stApp {background-color: #0E1117;}
    
    /* Login & UI */
    .login-container {position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 400px; padding: 40px; background: rgba(22, 27, 34, 0.9); border: 1px solid #30363d; border-radius: 16px; text-align: center;}
    .avatar-frame {display: flex; justify-content: center; margin-bottom: 15px;}
    .avatar-img {width: 100px; height: 100px; border-radius: 50%; object-fit: cover; border: 3px solid #58a6ff; box-shadow: 0 0 15px rgba(88, 166, 255, 0.5);}
    .stButton > button {border: 1px solid #30363d; font-weight: bold;}
    
    /* KPI & Trading */
    .kpi-card {background: #161b22; padding: 15px; border-radius: 6px; border: 1px solid #30363d; text-align: center;}
    .kpi-label {color: #8b949e; font-size: 12px; text-transform: uppercase; font-weight: 600; letter-spacing: 1px; margin-bottom: 5px;}
    .kpi-value {color: #f0f6fc; font-size: 24px; font-weight: 700;}
    
    /* Chat Style */
    .chat-container {height: 500px; overflow-y: auto; padding: 15px; background: #0d1117; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 15px; display: flex; flex-direction: column;}
    .msg-row {display: flex; margin-bottom: 10px; width: 100%;}
    .msg-mine {justify-content: flex-end;}
    .msg-theirs {justify-content: flex-start;}
    .bubble {padding: 10px 15px; border-radius: 15px; max-width: 70%; font-size: 14px; line-height: 1.4; word-wrap: break-word;}
    .bubble-mine {background: #1f6feb; color: white; border-bottom-right-radius: 2px;}
    .bubble-theirs {background: #21262d; color: #e6edf3; border-bottom-left-radius: 2px;}
    .msg-time {font-size: 10px; color: #8b949e; margin-top: 4px; display: block;}
    
    /* News Feed Style */
    .news-card {background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 15px; transition: 0.3s;}
    .news-card:hover {border-color: #58a6ff;}
    .news-title {font-size: 16px; font-weight: bold; color: #58a6ff; text-decoration: none;}
    .news-meta {font-size: 12px; color: #8b949e; margin-top: 5px;}
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION & IMPORTS ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_info' not in st.session_state: st.session_state['user_info'] = None
if 'chat_target' not in st.session_state: st.session_state['chat_target'] = None

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime
import feedparser # Thư viện lấy tin tức
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
# 4. DATABASE
# ==========================================
DB_FILE = "stable_cast.db"

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT, status TEXT)''')
    try: c.execute("ALTER TABLE users ADD COLUMN avatar TEXT"); 
    except: pass
    c.execute('''CREATE TABLE IF NOT EXISTS friends (user1 TEXT, user2 TEXT, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT, receiver TEXT, content TEXT, timestamp TEXT)''')
    try: c.execute("INSERT INTO users (username, password, role, status) VALUES (?, ?, ?, ?)", ("admin", hashlib.sha256("admin123".encode()).hexdigest(), "admin", "active")); conn.commit()
    except: pass
    conn.close()

init_db()

# --- HELPER FUNCTIONS (User, Auth, Social) ---
def image_to_base64(image):
    buffered = BytesIO(); image.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode()

def update_avatar(username, image_file):
    try:
        img = Image.open(image_file).resize((150, 150)); img_str = image_to_base64(img)
        conn = sqlite3.connect(DB_FILE); conn.execute("UPDATE users SET avatar=? WHERE username=?", (img_str, username)); conn.commit(); conn.close(); return True
    except: return False

def get_user_avatar(username):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor(); c.execute("SELECT avatar FROM users WHERE username=?", (username,)); data = c.fetchone(); conn.close(); return data[0] if data else None

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
    if not success: return False
    try:
        new_hash = hashlib.sha256(new_pass.encode()).hexdigest()
        conn = sqlite3.connect(DB_FILE); conn.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username)); conn.commit(); conn.close(); return True
    except: return False

def change_username(current_username, new_username):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor(); cursor.execute("SELECT * FROM users WHERE username=?", (new_username,))
        if cursor.fetchone(): conn.close(); return False
        conn.execute("UPDATE users SET username=? WHERE username=?", (new_username, current_username)); conn.commit(); conn.close(); return True
    except: return False

def get_all_users():
    conn = sqlite3.connect(DB_FILE); df = pd.read_sql("SELECT username, role, status FROM users", conn); conn.close(); return df

def update_user_status(u, act):
    conn = sqlite3.connect(DB_FILE)
    if act == "ban": conn.execute("UPDATE users SET status='banned' WHERE username=?", (u,))
    elif act == "unban": conn.execute("UPDATE users SET status='active' WHERE username=?", (u,))
    elif act == "delete": conn.execute("DELETE FROM users WHERE username=?", (u,))
    conn.commit(); conn.close()

# --- SOCIAL LOGIC ---
class SocialManager:
    @staticmethod
    def send_friend_request(from_u, to_u):
        if from_u == to_u: return "Self?"
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (to_u,))
        if not c.fetchone(): conn.close(); return "Not found"
        c.execute("SELECT * FROM friends WHERE (user1=? AND user2=?) OR (user1=? AND user2=?)", (from_u, to_u, to_u, from_u))
        if c.fetchone(): conn.close(); return "Exists"
        c.execute("INSERT INTO friends VALUES (?, ?, ?)", (from_u, to_u, "pending")); conn.commit(); conn.close(); return "Sent"

    @staticmethod
    def accept_friend(user, partner):
        conn = sqlite3.connect(DB_FILE)
        conn.execute("UPDATE friends SET status='accepted' WHERE user1=? AND user2=?", (partner, user))
        conn.commit(); conn.close()

    @staticmethod
    def get_friends(user):
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT user1, user2 FROM friends WHERE (user1=? OR user2=?) AND status='accepted'", (user, user))
        friends = []
        for row in c.fetchall(): friends.append(row[1] if row[0] == user else row[0])
        conn.close(); return friends

    @staticmethod
    def get_requests(user):
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT user1 FROM friends WHERE user2=? AND status='pending'", (user,))
        reqs = [r[0] for r in c.fetchall()]; conn.close(); return reqs

    @staticmethod
    def send_message(sender, receiver, content):
        if not content: return
        conn = sqlite3.connect(DB_FILE)
        ts = datetime.now().strftime("%H:%M")
        conn.execute("INSERT INTO messages (sender, receiver, content, timestamp) VALUES (?, ?, ?, ?)", (sender, receiver, content, ts))
        conn.commit(); conn.close()

    @staticmethod
    def get_messages(u1, u2):
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM messages WHERE (sender=? AND receiver=?) OR (sender=? AND receiver=?) ORDER BY id ASC", conn, params=(u1, u2, u2, u1))
        conn.close(); return df

# --- NEWS LOGIC ---
def get_crypto_news():
    try:
        # Lấy tin từ Cointelegraph RSS
        feed = feedparser.parse("https://cointelegraph.com/rss")
        return feed.entries[:10] # Lấy 10 tin mới nhất
    except:
        return []

# ==========================================
# 5. LOGIC CORE (AI + TRADE)
# ==========================================
class AIEngine:
    def __init__(self, look_back=60):
        self.look_back = look_back; self.scaler = MinMaxScaler((0,1)); self.close_scaler = MinMaxScaler((0,1)); self.model = None
    def prepare_data(self, df):
        df['SMA_50'] = df['close'].rolling(50).mean().fillna(method='bfill')
        df['SMA_200'] = df['close'].rolling(200).mean().fillna(method='bfill')
        data = df[['close', 'high', 'low', 'volume', 'RSI', 'ATR', 'MACD', 'Upper_Band', 'Lower_Band', 'SMA_50', 'SMA_200']].values
        self.close_scaler.fit(df[['close']]); return self.scaler.fit_transform(data)
    def build_model(self, input_shape):
        tf.random.set_seed(42); m = Sequential()
        m.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape)); m.add(Dropout(0.2))
        m.add(LSTM(32, return_sequences=False)); m.add(Dropout(0.2)); m.add(Dense(16, activation='relu')); m.add(Dense(1)); m.compile(optimizer='adam', loss='mse'); return m
    def train_and_predict(self, df, epochs=30):
        scaled = self.prepare_data(df); X, y = [], []
        for i in range(self.look_back, len(scaled)): X.append(scaled[i-self.look_back:i]); y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        if 'ai_model' not in st.session_state:
            self.model = self.build_model((self.look_back, 11)); self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=False, callbacks=[EarlyStopping('loss', patience=5)])
            st.session_state['ai_model'] = self.model
        else: self.model = st.session_state['ai_model']
        last_seq = scaled[-self.look_back:].reshape(1, self.look_back, 11); pred = self.model.predict(last_seq)
        return self.close_scaler.inverse_transform(pred)[0][0]

class TradeManager:
    FILE_NAME = "trade_history_v35.csv"
    COOLDOWN_SECONDS = 180
    @staticmethod
    def init_file():
        if not os.path.exists(TradeManager.FILE_NAME): pd.DataFrame(columns=["timestamp", "symbol", "type", "entry", "tp", "sl", "status", "confidence", "user"]).to_csv(TradeManager.FILE_NAME, index=False)
    @staticmethod
    def reset_history():
        if os.path.exists(TradeManager.FILE_NAME): os.remove(TradeManager.FILE_NAME); TradeManager.init_file()
    @staticmethod
    def log_trade(symbol, trade_type, entry, tp, sl, conf, user, discord_url=None, tp1=None, be_level=None):
        TradeManager.init_file(); df = pd.read_csv(TradeManager.FILE_NAME)
        active = df[(df['symbol'] == symbol) & (df['status'] == 'PENDING') & (df['user'] == user)]
        if not active.empty: return "PENDING"
        user_history = df[(df['symbol'] == symbol) & (df['user'] == user)]
        if not user_history.empty:
            try:
                if (datetime.now() - datetime.strptime(user_history.iloc[-1]['timestamp'], "%Y-%m-%d %H:%M")).total_seconds() < TradeManager.COOLDOWN_SECONDS: return "COOLDOWN"
            except: pass
        new_row = pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "symbol": symbol, "type": trade_type, "entry": entry, "tp": tp, "sl": sl, "status": "PENDING", "confidence": conf, "user": user}])
        df = pd.concat([df, new_row], ignore_index=True); df.to_csv(TradeManager.FILE_NAME, index=False)
        if discord_url:
            color = 3447003 if "LONG" in trade_type else 15158332
            requests.post(discord_url, json={"embeds": [{"title": f"🛡️ SAFE SIGNAL: {symbol}", "description": f"**Trader:** {user}\n**AI Conf:** {conf:.1f}%", "color": color, "fields": [{"name": "Action", "value": f"**{trade_type}** @ `${entry:,.2f}`", "inline": False},{"name": "Targets", "value": f"🎯 **TP1:** `${tp1:,.2f}`\n🚀 **TP2:** `${tp:,.2f}`", "inline": True},{"name": "Safety", "value": f"🛑 **SL:** `${sl:,.2f}`\n⚠️ **Move SL:** `${be_level:,.2f}`", "inline": True}]}]})
        return "SUCCESS"
    @staticmethod
    def audit_trades(market_df, symbol):
        TradeManager.init_file()
        try:
            df = pd.read_csv(TradeManager.FILE_NAME); df['timestamp'] = pd.to_datetime(df['timestamp']); updated = False
            if df.empty: return 0.0, df
            return 0.0, df 
        except: return 0.0, pd.DataFrame()

def calculate_confidence(df, direction):
    last = df.iloc[-1]; score = 50.0; rsi = last['RSI']
    sma50 = last.get('SMA_50', last['close']); sma200 = last.get('SMA_200', last['close'])
    if direction == "LONG":
        if rsi < 30: score += (30 - rsi)*1.5
        if last['close'] > sma50: score += 10
        if sma50 > sma200: score += 5
    elif direction == "SHORT":
        if rsi > 70: score += (rsi - 70)*1.5
        if last['close'] < sma50: score += 10
        if sma50 < sma200: score += 5
    return max(15.0, min(99.9, score))

def background_bot_logic(symbol, webhook_url):
    while True:
        try:
            CryptoDataCollector.fetch_and_save_data(); df = pd.read_csv(f"{symbol}_data.csv")
            df['SMA_50'] = df['close'].rolling(50).mean(); df['SMA_200'] = df['close'].rolling(200).mean()
            last = df.iloc[-1]; signal = None
            if last['RSI'] < 30: signal = "LONG"
            elif last['RSI'] > 70: signal = "SHORT"
            if signal and webhook_url:
                conf = calculate_confidence(df, signal)
                if not os.path.exists("trade_history_v35.csv"): pd.DataFrame(columns=["status"]).to_csv("trade_history_v35.csv", index=False)
                df_hist = pd.read_csv("trade_history_v35.csv")
                active = df_hist[(df_hist['symbol'] == symbol) & (df_hist['status'] == 'PENDING')]
                if active.empty:
                    atr = last['ATR']; tp = last['close'] + (2.5 * atr) if signal == "LONG" else last['close'] - (2.5 * atr); sl = last['close'] - (1.2 * atr) if signal == "LONG" else last['close'] + (1.2 * atr); color = 3447003 if signal == "LONG" else 15158332
                    requests.post(webhook_url, json={"embeds": [{"title": f"🔔 URGENT BOT: {symbol}", "description": f"**Signal:** {signal}\n**Confidence:** {conf:.1f}%", "color": color, "fields": [{"name": "Entry", "value": f"${last['close']:,.2f}", "inline": True}, {"name": "TP / SL", "value": f"${tp:,.2f} / ${sl:,.2f}", "inline": True}]}]})
            time.sleep(900)
        except: time.sleep(60)

@st.cache_resource
def start_background_thread(symbol, webhook):
    t = threading.Thread(target=background_bot_logic, args=(symbol, webhook), daemon=True); t.start()

# ==========================================
# 6. UI RENDERER (MAIN)
# ==========================================
main_container = st.empty()
SYSTEM_WEBHOOK = "https://discord.com/api/webhooks/1469612104616251561/SvDfdD1c3GF4evKxTcLCvXGQtPrxrWQBK1BgcpCDh59olo6tQD1zb7ENNHGiFaE0JoBR"

def render_login():
    with main_container.container():
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<div style='text-align:center; padding-top:50px;'><h1 style='color:#fff;'>STABLE CAST AI</h1></div>", unsafe_allow_html=True)
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

# --- TRANG: NEWS FEED ---
def render_news_page():
    st.header("📰 MARKET NEWS")
    st.markdown("Tin tức nóng hổi từ Cointelegraph (Cập nhật liên tục)")
    
    if st.button("🔄 Refresh News"): st.rerun()
    
    news = get_crypto_news()
    if not news:
        st.warning("Không tải được tin tức lúc này. Vui lòng thử lại sau.")
        return

    for item in news:
        st.markdown(f"""
        <div class="news-card">
            <a href="{item.link}" target="_blank" class="news-title">{item.title}</a>
            <div class="news-meta">{item.published}</div>
            <div style="font-size:14px; color:#ccc; margin-top:5px;">{item.summary.split('<')[0]}...</div>
        </div>
        """, unsafe_allow_html=True)

# --- TRANG: CHAT SOCIAL ---
def render_social_page(user):
    st.header("💬 CHAT ROOM")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Friends")
        with st.expander("➕ Add Friend"):
            target_u = st.text_input("Username", label_visibility="collapsed")
            if st.button("Send Request"):
                res = SocialManager.send_friend_request(user, target_u)
                if res == "Sent": st.success("Sent!")
                else: st.error(res)
        
        reqs = SocialManager.get_requests(user)
        if reqs:
            st.info(f"{len(reqs)} Requests")
            for r in reqs:
                if st.button(f"Accept {r}", key=f"acc_{r}"):
                    SocialManager.accept_friend(user, r); st.rerun()
        
        friends = SocialManager.get_friends(user)
        st.write("---")
        for f in friends:
            if st.button(f"🟢 {f}", key=f"chat_{f}", use_container_width=True):
                st.session_state['chat_target'] = f; st.rerun()

    with c2:
        target = st.session_state['chat_target']
        if target:
            st.subheader(f"Chatting with {target}")
            msgs = SocialManager.get_messages(user, target)
            
            # Chat Container
            html = "<div class='chat-container'>"
            for _, row in msgs.iterrows():
                cls = "msg-mine" if row['sender'] == user else "msg-theirs"
                bbl = "bubble-mine" if row['sender'] == user else "bubble-theirs"
                html += f"<div class='msg-row {cls}'><div class='bubble {bbl}'>{row['content']}<span class='msg-time'>{row['timestamp']}</span></div></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
            
            with st.form("chat"):
                txt = st.text_input("Message", label_visibility="collapsed")
                if st.form_submit_button("Send 🚀") and txt:
                    SocialManager.send_message(user, target, txt); st.rerun()
            if st.button("Refresh Chat"): st.rerun()
        else: st.info("👈 Select a friend to start chatting")

# --- UI MAIN NAVIGATION ---
def render_dashboard():
    main_container.empty()
    user, role = st.session_state['user_info']
    avatar_b64 = get_user_avatar(user)
    img_src = f"data:image/png;base64,{avatar_b64}" if avatar_b64 else "https://cdn-icons-png.flaticon.com/512/847/847969.png"
    
    with st.sidebar:
        st.markdown(f"<div class='avatar-frame'><img src='{img_src}' class='avatar-img'></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{user}</h3>", unsafe_allow_html=True)
        
        # --- MENU 3 MÓN ---
        page_mode = st.radio("MENU", ["📈 Trading", "💬 Chat Room", "📰 Market News"], label_visibility="collapsed")
        st.write("---")
        
        with st.expander("👤 Profile"):
            tab_ava, tab_name, tab_sec = st.tabs(["Avatar", "Name", "Sec"])
            with tab_ava:
                uploaded = st.file_uploader("Img", type=['png','jpg'])
                if uploaded and update_avatar(user, uploaded): st.success("OK"); time.sleep(1); st.rerun()
            with tab_name:
                new_u = st.text_input("New Name")
                if st.button("Rename"):
                    if change_username(user, new_u): st.session_state['logged_in']=False; st.rerun()
            with tab_sec:
                old_p = st.text_input("Old Pass", type="password")
                new_p = st.text_input("New Pass", type="password")
                if st.button("Update"): 
                    if change_password(user, old_p, new_p): st.success("OK")
        
        if st.button("Sign Out"): st.session_state['logged_in'] = False; st.rerun()
        if role == 'admin' and st.checkbox("🛡️ Admin"):
            st.dataframe(get_all_users(), hide_index=True)
            return

    # --- ĐIỀU HƯỚNG TRANG ---
    if page_mode == "💬 Chat Room":
        render_social_page(user)
        return
    
    if page_mode == "📰 Market News":
        render_news_page()
        return

    # --- TRADING PAGE ---
    with st.sidebar:
        st.header("💎 ASSETS")
        symbol = st.selectbox("Select", ["BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "PAXG_USDT"])
        wb = SYSTEM_WEBHOOK 
        if role == 'admin':
            wb = st.text_input("Webhook", value=SYSTEM_WEBHOOK, type="password")
            if st.button("Bot 24/7"): start_background_thread(symbol, wb)
        if st.button("Update Data"):
            with st.spinner("AI V35 Thinking..."): 
                CryptoDataCollector.fetch_and_save_data()
                if 'ai_model' in st.session_state: del st.session_state['ai_model']
            st.rerun()

    try: df = pd.read_csv(f"{symbol}_data.csv"); df['timestamp'] = pd.to_datetime(df['timestamp']); df.sort_values('timestamp', inplace=True)
    except: st.info("Need Update."); return

    win, history = TradeManager.audit_trades(df, symbol)
    last = df.iloc[-1]; color = "#00E676"
    
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
            eng = AIEngine(60); pred = eng.train_and_predict(df, 30)
            direct = "LONG" if pred > last['close'] else "SHORT"
            conf = calculate_confidence(df, direct)
            safe = (direct=="LONG" and last['RSI']<70) or (direct=="SHORT" and last['RSI']>30) or (conf > 70)
            col_sig = "#00E676" if direct == "LONG" else "#FF5252"
            
            if safe:
                atr = last['ATR']
                if direct == "LONG": tp=last['close']+(2.5*atr); sl=last['close']-(1.2*atr); tp1=last['close']+(1.2*atr); be_level=last['close']+(0.8*atr)
                else: tp=last['close']-(2.5*atr); sl=last['close']+(1.2*atr); tp1=last['close']-(1.2*atr); be_level=last['close']-(0.8*atr)
                conf_color = "#00E676" if conf > 80 else "#FFD700" if conf > 50 else "#FF5252"
                
                html = f"""
<div style="background:#0d1117; border-radius:8px; border:1px solid #30363d; padding:20px; border-top: 3px solid {col_sig}">
<div style="font-size:32px; font-weight:800; color:{col_sig}; text-align:center;">{direct}</div>
<div style="text-align:center; color:#888; font-size:12px;">AI Target: ${pred:,.2f}</div>
<div style="margin:15px 0; border:1px dashed #FFD700; background:rgba(255,215,0,0.05); padding:10px; border-radius:5px;">
    <div style="font-size:11px; color:#FFD700; font-weight:bold; margin-bottom:5px; text-align:center;">🛡️ SAFE PLAN</div>
    <div style="display:flex; justify-content:space-between; font-size:12px;"><span style="color:#aaa">TP1:</span><span style="color:#fff">${tp1:,.2f}</span></div>
    <div style="display:flex; justify-content:space-between; font-size:12px;"><span style="color:#aaa">Move SL at:</span><span style="color:#fff">${be_level:,.2f}</span></div>
</div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">ENTRY</span><span style="color:#fff">${last['close']:,.2f}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">TP2</span><span style="color:#00E676">${tp:,.2f}</span></div>
<div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px 0; font-size:13px;"><span style="color:#888">SL</span><span style="color:#FF5252">${sl:,.2f}</span></div>
</div>
"""
                ai.markdown(html, unsafe_allow_html=True)
                if st.button("🚀 PUSH TO DISCORD", use_container_width=True):
                    status = TradeManager.log_trade(symbol, direct, last['close'], tp, sl, conf, user, wb, tp1, be_level)
                    if status=="SUCCESS": st.toast("Sent!", icon="🚀")
                    elif status=="PENDING": st.toast("Pending active!", icon="🚫")
                    elif status=="COOLDOWN": st.toast("Wait 3m!", icon="⏱️")
            else:
                ai.markdown(f"""<div style="background:#0d1117; padding:20px; text-align:center; border-radius:8px; opacity:0.6"><h3>NO TRADE</h3><p>Risk High</p></div>""", unsafe_allow_html=True)
        except Exception as e: ai.error(f"Error: {e}")

if st.session_state['logged_in']: render_dashboard()
else: render_login()
