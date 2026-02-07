import ccxt
import pandas as pd
import time
import os
from datetime import datetime

# --- CẤU HÌNH ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'PAXG/USDT'] 
TIMEFRAME = '1h'
LIMIT = 1000

def fetch_and_save_data():
    exchange = ccxt.binance({'enableRateLimit': True})
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ Đang lấy dữ liệu NÂNG CAO (MACD + BB)...")
    
    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LIMIT)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 1. RSI (Sức mạnh tương đối)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 2. ATR (Độ biến động)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()

            # 3. MACD (Chỉ báo xu hướng - MỚI)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 4. Bollinger Bands (Dải băng - MỚI)
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['Std_Dev'] = df['close'].rolling(window=20).std()
            df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
            df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
            
            df.dropna(inplace=True) # Xóa dữ liệu NaN do tính toán
            
            filename = f"{symbol.replace('/', '_')}_data.csv"
            df.to_csv(filename, index=False)
            print(f"   ✅ Đã cập nhật: {symbol} (Đủ RSI, ATR, MACD, BB)")
            
        except Exception as e:
            print(f"   ❌ Lỗi {symbol}: {e}")

if __name__ == "__main__":
    while True:
        fetch_and_save_data()
        print("   💤 Chờ 1 tiếng nữa...")
        now = datetime.now()
        sleep_seconds = 3600 - (now.minute * 60 + now.second)
        time.sleep(sleep_seconds)