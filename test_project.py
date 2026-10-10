import cv2
import csv
import time
import os
from collections import defaultdict
from ultralytics import YOLO

videos = [
    ("day", "input/car.mp4", [2, 5, 7], 0.4, 640),
    ("night", "input/night.mp4", [0, 2, 5, 7], 0.1, 1280)
]

os.makedirs("test_results", exist_ok=True)

with open("night_test_tracker.yaml", "w") as f:
    f.write(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.3\n"
        "track_buffer: 60\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )

summary = []

for name, path, classes, conf, imgsz in videos:

    if not os.path.exists(path):
        print("Файл не найден:", path)
        continue

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Не удалось открыть:", path)
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1280 / w, 720 / h, 1)

    model = YOLO("yolov8n.pt")
    counts = defaultdict(int)
    ids = defaultdict(set)
    start = time.perf_counter()
    frame_num = 0

    print(f"\nТестирую {path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if name == "night":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_input = cv2.cvtColor(
                cv2.merge((clahe.apply(l), a, b)),
                cv2.COLOR_LAB2BGR
            )
        else:
            frame_input = frame

        result = model.track(
            frame_input,
            persist=True,
            tracker="night_test_tracker.yaml" if name == "night" else "bytetrack.yaml",
            classes=classes,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            cls = result.boxes.cls.cpu().numpy().astype(int)

            for c in cls:
                counts[c] += 1

            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for c, obj_id in zip(cls, track_ids):
                    ids[c].add(int(obj_id))

        if total and frame_num in [max(1, int(total * p)) for p in (0.25, 0.5, 0.75)]:
            base = f"test_results/{name}_{frame_num}"
            cv2.imwrite(base + "_original.jpg", frame)
            cv2.imwrite(base + "_detected.jpg", result.plot())

        if frame_num % 50 == 0:
            print(f"Обработано кадров: {frame_num}/{total}")

    elapsed = time.perf_counter() - start
    cap.release()

    row = {
        "video": path,
        "frames": frame_num,
        "processing_seconds": round(elapsed, 2),
        "processing_fps": round(frame_num / elapsed, 2) if elapsed else 0,
        "detections": sum(counts.values()),
        "unique_track_ids": sum(len(v) for v in ids.values())
    }

    for c, label in [(0, "person"), (2, "car"), (5, "bus"), (7, "truck")]:
        row[label + "_detections"] = counts[c]
        row[label + "_ids"] = len(ids[c])

    summary.append(row)

    print("Время обработки:", row["processing_seconds"], "сек.")
    print("Скорость обработки:", row["processing_fps"], "кадров/сек.")
    print("Всего обнаружений:", row["detections"])
    print("Уникальных ID:", row["unique_track_ids"])

if summary:
    with open("test_results/summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

print("\nТестирование завершено. Результаты в папке test_results.")