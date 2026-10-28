from ultralytics import YOLO
import cv2

# Tải model đã train
model = YOLO("best.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được webcam!")
    exit()

print("✅ Webcam đã mở!")
print("Nhấn Q để thoát.")

while True:
    # Đọc hình ảnh từ webcam
    success, frame = cap.read()

    if not success:
        print("❌ Không đọc được hình ảnh từ webcam!")
        break

    # YOLO nhận diện
    results = model(frame, conf=0.5)

    # Vẽ kết quả nhận diện lên hình
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("AI Nhan Dien Sinh Vat bien - Con Dao", annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Đóng webcam
cap.release()
cv2.destroyAllWindows()