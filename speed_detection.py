import cv2
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

video_path = "input/car.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Видео не открылось.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# Точки дороги
src_points = np.float32([
    [265, 415],
    [479, 414],
    [1153, 476],
    [1005, 602]
])

# Размер дорожного участка
dst_points = np.float32([
    [0, 0],
    [3.5, 0],
    [3.5, 20],
    [0, 20]
])

matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)

# История положения машин
history = defaultdict(lambda: deque(maxlen=10))

# Предыдущий кадр
previous_gray = None

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # классический метод: gaussian blur

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # классический метод: threshold + contours

    if previous_gray is not None:

        difference = cv2.absdiff(
            previous_gray,
            blur
        )

        _, motion_mask = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Поиск областей движения
        moving_objects = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 300:
                moving_objects.append(contour)

    previous_gray = blur

    # yolo + traking

    results = model.track(
        frame,
        persist=True,
        classes=[2, 5, 7],
        conf=0.4
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()

        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Нижняя центральная точка автомобиля
            center_x = int((x1 + x2) / 2)
            bottom_y = y2

            # перевод пикселей в метры

            point = np.float32([
                [[center_x, bottom_y]]
            ])

            real_point = cv2.perspectiveTransform(
                point,
                matrix
            )[0][0]

            real_x = real_point[0]
            real_y = real_point[1]

            # Сохраняем координаты
            history[vehicle_id].append(
                (frame_number, real_x, real_y)
            )

            speed = 0

            # расчет скорости

            if len(history[vehicle_id]) >= 8:

                old_frame, old_x, old_y = history[vehicle_id][0]

                distance = math.sqrt(
                    (real_x - old_x) ** 2 +
                    (real_y - old_y) ** 2
                )

                time = (
                    frame_number - old_frame
                ) / fps

                if time > 0:

                    speed = (
                        distance / time
                    ) * 3.6

            # убираю неправильные значения
            if speed > 120:
                speed = 0

            # вывод

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {vehicle_id}  {speed:.1f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Vehicle Speed",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()