import cv2
from ultralytics import YOLO
import os
import torch

import os

# --- Base directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Paths (relative to script) ---
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample1.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "runs", "inference", "helmet_local_output.mp4")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Open Video ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video: {VIDEO_PATH}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

WINDOW_NAME = "Helmet Detection (Local Model)"
cv2.namedWindow(WINDOW_NAME)
frame_delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]  # get detection results

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Only show helmet/no_helmet labels
            if label.lower() in ["helmet", "no_helmet"]:
                color = (0, 255, 0) if label.lower() == "helmet" else (0, 0, 255)
                text = f"{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # optional: small circle at center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 3, color, -1)

    cv2.imshow(WINDOW_NAME, frame)
    out.write(frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"📁 Helmet detection output saved at: {OUTPUT_PATH}")

