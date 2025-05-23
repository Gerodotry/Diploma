from ultralytics import YOLO

model = YOLO('/runs/detect/train/weights/best.pt')

# Вивід списку всіх класів, які знає модель
print("📋 Класи моделі:")
for cls_id, name in model.names.items():
    print(f"{cls_id}: {name}")
