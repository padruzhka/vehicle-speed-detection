import cv2
import os
import numpy as np

videos = [
    ("day", "input/car.mp4"),
    ("night", "input/night.mp4")
]

os.makedirs("test_results", exist_ok=True)

for name, path in videos:

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Не удалось открыть:", path)
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, total // 4)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret1, previous = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, min(start + 10, total - 1))
    ret2, frame = cap.read()

    cap.release()

    if not ret1 or not ret2:
        print("Не удалось прочитать кадры:", name)
        continue

    scale = min(960 / frame.shape[1], 540 / frame.shape[0], 1)
    size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))

    frame = cv2.resize(frame, size)
    previous = cv2.resize(previous, size)

    # 1. Исходный кадр
    cv2.imwrite(f"test_results/methods_{name}_original.jpg", frame)

    # 2. Перевод в оттенки серого и Gaussian Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    cv2.imwrite(f"test_results/methods_{name}_blur.jpg", blur)

    # 3. Разница кадров и бинарная маска
    difference = cv2.absdiff(prev_blur, blur)

    _, mask = cv2.threshold(
        difference, 30, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(f"test_results/methods_{name}_mask.jpg", mask)

    # 4. Контуры движущихся областей
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = frame.copy()
    found = 0

    for contour in contours:
        if cv2.contourArea(contour) > 150:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            found += 1

    cv2.imwrite(f"test_results/methods_{name}_contours.jpg", result)

    print(f"{name}: найдено областей движения: {found}")

print("Готово")