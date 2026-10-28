# Côn Đảo Conservation — bản web

## Các file cần có
- `app.py`: giao diện web Streamlit, thay cho giao diện Tkinter.
- `requirements.txt`: thư viện cần cài.
- `best.pt`: file mô hình YOLO của dự án gốc (bạn cần tự thêm).
- `assests/`: thư mục ảnh gốc, nếu muốn hiện ảnh ở mục Giới thiệu (giữ đúng tên thư mục hiện tại).

## Chạy thử trên máy Windows
1. Cài Python 3.10 hoặc 3.11.
2. Mở Terminal tại thư mục chứa `app.py` và chạy:
   `pip install -r requirements.txt`
3. Đặt API key Gemini trong PowerShell cho phiên terminal hiện tại:
   `$env:GEMINI_API_KEY="API_KEY_CUA_BAN"`
4. Chạy:
   `streamlit run app.py`

## Đưa lên Streamlit Community Cloud
1. Đẩy `app.py`, `requirements.txt`, `best.pt` và thư mục `assests` lên repository GitHub.
2. Vào https://share.streamlit.io/ và kết nối repository.
3. Chọn `app.py` làm Main file path rồi Deploy.
4. Trong App settings → Secrets, thêm:
   `GEMINI_API_KEY = "API_KEY_CUA_BAN"`

Không commit API key thật vào GitHub. Nếu `best.pt` quá lớn hoặc máy chủ miễn phí thiếu bộ nhớ, cần dùng máy chủ phù hợp hơn hoặc lưu model ở nơi riêng và tải model lúc chạy.

## Ghi chú
- Bản này chuyển giao diện từ Tkinter sang Streamlit và giữ luồng chính YOLO → Gemini.
- Phần đăng nhập ở giao diện Tkinter gốc chỉ là nút giao diện, chưa có cơ chế xác thực người dùng thực sự.
- Nội dung Giới thiệu được dựng lại ở mức khái quát; nếu muốn giữ nguyên toàn bộ nội dung/hình ảnh nhúng trong file gốc, cần tách và chuyển riêng dữ liệu đó.
