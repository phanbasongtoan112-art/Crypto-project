import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import os

SYMBOLS = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT', 'PAXG_USDT']
EPOCHS = 50       
LOOK_BACK = 60 # Bạn có thể thử tăng lên 168 (1 tuần) để xem nó có bớt trễ không
PATIENCE = 5      

def train_model(symbol):
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG HUẤN LUYỆN: {symbol} (VỚI MACD & BOLLINGER BANDS)")
    print(f"{'='*50}")
    
    file_path = f"{symbol}_data.csv"
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file. Hãy chạy Collector lại!")
        return

    df = pd.read_csv(file_path)
    
    # --- QUAN TRỌNG: Thêm các cột dữ liệu mới vào AI ---
    # Bây giờ AI sẽ nhìn vào 9 yếu tố thay vì 6 như trước
    features = ['close', 'high', 'low', 'volume', 'RSI', 'ATR', 'MACD', 'Upper_Band', 'Lower_Band']
    
    # Kiểm tra xem file data đã có cột mới chưa (tránh lỗi nếu chưa chạy Collector mới)
    if 'MACD' not in df.columns:
        print("❌ File dữ liệu cũ chưa có MACD/BB. Vui lòng chạy lại CryptoDataCollector.py trước!")
        return

    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(df[['close']])

    X, y = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X.append(scaled_data[i-LOOK_BACK:i])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential()
    # Tăng neuron lên 128 để não to hơn
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mse')

    print(f"⏳ Bắt đầu học tối đa {EPOCHS} vòng...")
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    print("\n🔍 Đang vẽ biểu đồ...")
    preds = model.predict(X_test)
    preds_price = close_scaler.inverse_transform(preds)
    actual_price = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    mape = np.mean(np.abs((actual_price - preds_price) / actual_price)) * 100
    
    print(f"✅ HOÀN TẤT! Sai số (MAPE): {mape:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual_price, color='blue', label='Thực Tế')
    plt.plot(preds_price, color='red', label='AI Dự Đoán (Đã cải tiến)')
    plt.title(f"MODEL NÂNG CAO: {symbol} - Sai số: {mape:.2f}%")
    plt.xlabel('Thời gian')
    plt.ylabel('Giá')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    while True:
        print("\n🤖 PRO AI TRAINING MENU")
        for i, sym in enumerate(SYMBOLS): print(f"  {i+1}. {sym}")
        print("  0. All")
        try:
            c = int(input("👉 Chọn: "))
            if c == 0:
                for s in SYMBOLS: train_model(s)
                break
            elif 1 <= c <= len(SYMBOLS):
                train_model(SYMBOLS[c-1])
                if input("Train tiếp? (y/n): ").lower() != 'y': break
        except: print("Lỗi nhập!")