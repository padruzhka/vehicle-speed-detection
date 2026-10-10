import cv2
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

video_path = "input/night.mp4"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Видео не открылось.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Размер видео
w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = min(1280 / w0, 720 / h0, 1)
w, h = int(w0 * scale), int(h0 * scale)

# Точки ночной дороги
src_points = np.float32([
    [918, 97],
    [510, 206],
    [852, 666],
    [1208, 470]
])

dst_points = np.float32([
    [0, 0],
    [3.5, 0],
    [3.5, 20],
    [0, 20]
])

matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Настройки отслеживания
with open("night_tracker.yaml", "w") as f:
    f.write(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.3\n"
        "track_buffer: 60\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )

history = defaultdict(lambda: deque(maxlen=15))
point_history = defaultdict(lambda: deque(maxlen=5))
speed_history = defaultdict(lambda: deque(maxlen=3))

previous_gray = None
frame_number = 0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    frame = cv2.resize(frame, (w, h))

    # Gaussian Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold + Contours
    if previous_gray is not None:
        difference = cv2.absdiff(previous_gray, blur)
        _, mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        moving_objects = [c for c in contours if cv2.contourArea(c) > 300]

    previous_gray = blur

    # Улучшение контраста ночного видео
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR
    )

    # YOLO + Tracking
    results = model.track(
        enhanced, persist=True, tracker="night_tracker.yaml",
        classes=[0, 2, 5, 7], conf=0.1, imgsz=1280,
        iou=0.5, verbose=False
    )

    boxes = results[0].boxes

    if boxes.id is not None:
        coordinates = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, obj_id, cls in zip(coordinates, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            # Пешеход
            if cls == 0:
                color = (255, 0, 0)
                label = f"Person ID {obj_id}"

            else:
                color = (0, 255, 0)

                # Нижняя центральная точка машины
                point_history[obj_id].append(((x1 + x2) / 2, y2))
                center_x = np.median([p[0] for p in point_history[obj_id]])
                bottom_y = np.median([p[1] for p in point_history[obj_id]])

                point = np.float32([[[center_x, bottom_y]]])
                real_x, real_y = cv2.perspectiveTransform(point, matrix)[0][0]

                history[obj_id].append((frame_number, real_x, real_y))
                speed = 0.0

                # Расчёт скорости
                if len(history[obj_id]) >= 8:
                    old_frame, old_x, old_y = history[obj_id][0]

                    distance = math.hypot(real_x - old_x, real_y - old_y)
                    elapsed_time = (frame_number - old_frame) / fps

                    if elapsed_time > 0:
                        speed = distance / elapsed_time * 3.6

                    if distance < 0.15 or speed < 1.5 or speed > 120:
                        speed = 0.0

                    speed_history[obj_id].append(speed)
                    speed = float(np.median(speed_history[obj_id]))

                label = f"ID {obj_id} | {speed:.1f} km/h"

            # Вывод результата
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

    # Отображение видео
    display_scale = min(960 / w, 540 / h, 1)
    display = cv2.resize(
        frame, (int(w * display_scale), int(h * display_scale))
    )

    cv2.imshow("Night Vehicle Speed", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Обработка ночного видео завершена.")