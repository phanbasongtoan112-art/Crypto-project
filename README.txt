ĐỒ ÁN MÔN HỌC: ADY201m
NHÓM THỰC HIỆN: TEAM 1
Sinh viên: Phan Bá Song Toàn (Trưởng nhóm)

TÊN DỰ ÁN: Crypto Currency Price Prediction
Hệ thống AI Trading & Quản trị rủi ro Crypto thời gian thực (Real-time Crypto Trading System)

1. GIỚI THIỆU
   Đây là hệ thống Dashboard hỗ trợ ra quyết định giao dịch dựa trên dữ liệu thật (Real-time) từ Binance.
   Hệ thống kết hợp Deep Learning (LSTM) để dự báo xu hướng và ATR (Average True Range) để tính toán điểm Stop Loss/Take Profit tự động.

2. CÔNG NGHỆ SỬ DỤNG
   - Ngôn ngữ: Python
   - Giao diện: Streamlit
   - AI Model: TensorFlow/Keras (Bi-LSTM)
   - Data Source: CCXT (Binance API)
   - Visualization: Plotly Interactive Charts

3. HƯỚNG DẪN CÀI ĐẶT & CHẠY
   Bước 1: Cài thư viện
     pip install -r requirements.txt

   Bước 2: Chạy hệ thống (Mở 2 Terminal song song)
     Terminal 1 (Data Collector): python CryptoDataCollector.py
     Terminal 2 (Web Dashboard):  python -m streamlit run Dashboard.py

4. TÍNH NĂNG CHÍNH
   - Auto-Update: Dữ liệu tự động cập nhật mỗi giờ (hoặc bấm nút Force Update).
   - Signal Engine: Tự động đưa ra tín hiệu MUA/BÁN.
   - Risk Management: Tự động tính R:R (Risk/Reward) = 1:2.