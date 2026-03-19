from ultralytics import YOLO
import time

# 1. Ładowanie najnowszego modelu YOLOv26 Nano
# Ten model jest zoptymalizowany pod kątem szybkości na CPU (idealny na RPi)
model = YOLO('yolo26n.pt')

# 2. Test na standardowym obrazku
source_img = 'https://ultralytics.com/images/bus.jpg'

print("Rozpoczynam detekcję na YOLOv26n...")
start_time = time.time()

# Stream=True pomaga przy oszczędzaniu RAMu na Raspberry Pi
results = model.predict(source=source_img, save=True, conf=0.3, stream=False)

end_time = time.time()
print(f"Pełny czas operacji (ładowanie + detekcja): {round(end_time - start_time, 3)}s")

# 3. Wyciągamy konkretne nazwy obiektów
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Pobieramy nazwę klasy z słownika modelu
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        print(f"--> Wykryto: {class_name} ({round(confidence * 100, 1)}%)")

print("\nGotowe! Sprawdź folder 'runs/detect', żeby zobaczyć wynikowy plik.")
