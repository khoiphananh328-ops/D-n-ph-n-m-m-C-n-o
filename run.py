from pathlib import Path
from ultralytics import YOLO

# Lấy thư mục chứa file check_model.py
project_folder = Path(__file__).resolve().parent

# Tìm best.pt trong cùng thư mục
model_path = project_folder / "best.pt"

print("Đang tìm model tại:")

print(model_path)

if not model_path.exists():
    print("❌ Không tìm thấy best.pt!")
    exit()

print("\nĐang tải model...")

model = YOLO(str(model_path))

print("\n✅ MODEL ĐÃ TẢI THÀNH CÔNG!")
print("Các loài đã được train:")

for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")
    image_dir = project_folder / "test"

image_files = (
    list(image_dir.glob("*.jpg")) +
    list(image_dir.glob("*.jpeg")) +
    list(image_dir.glob("*.png"))
)

if not image_files:
    print("❌ Không tìm thấy ảnh trong thư mục test")
    exit()

image_path = image_files[0]

print("Ảnh đang nhận diện:")
print(image_path)

results = model(str(image_path))

for result in results:
    result.show()

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(f"Phát hiện: {class_name} | Độ tin cậy: {confidence:.2%}")