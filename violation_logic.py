import cv2
import os
import shutil
from datetime import datetime
from ultralytics import YOLO
import math
from tkinter import Tk, filedialog

import os

# -------------------------
# Base directory (script location)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Paths (relative to script)
# -------------------------
TRAFFIC_MODEL_PATH = os.path.join(BASE_DIR, "src", "yolo11n.pt")
HELMET_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DETECTED_DIR = os.path.join(BASE_DIR, "detected_violations")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HELMET_DIR = os.path.join(DETECTED_DIR, "helmet")
TRAFFIC_DIR = os.path.join(DETECTED_DIR, "traffic")


# -------------------------
# Cleanup old detections
# -------------------------
for folder in [HELMET_DIR, TRAFFIC_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# -------------------------
# Load YOLO models
# -------------------------
traffic_model = YOLO(TRAFFIC_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)

# -------------------------
# Dynamic video selection
# -------------------------
root = Tk()
root.withdraw()
VIDEO_PATH = filedialog.askopenfilename(title="Select Video",
                                        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
if not VIDEO_PATH:
    print("❌ No video selected.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
delay = int(1000 / fps)

ret, first_frame = cap.read()
if not ret:
    print("❌ Cannot read first frame.")
    exit()

# -------------------------
# Line drawing
# -------------------------
line_points = []

def draw_line(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        line_points.clear()

cv2.namedWindow("Draw Line")
cv2.setMouseCallback("Draw Line", draw_line)

while True:
    frame_copy = first_frame.copy()
    for pt in line_points:
        cv2.circle(frame_copy, pt, 5, (0, 255, 0), -1)
    if len(line_points) == 2:
        cv2.line(frame_copy, line_points[0], line_points[1], (0, 255, 0), 2)
    cv2.putText(frame_copy, "Draw 2 points for boundary, Press ENTER to start",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Draw Line", frame_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(line_points) == 2:  # ENTER
        break
    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw Line")

x1_line, y1_line = line_points[0]
x2_line, y2_line = line_points[1]
line_x_min, line_x_max = min(x1_line, x2_line), max(x1_line, x2_line)
line_y_avg = (y1_line + y2_line) // 2

# -------------------------
# Helper function
# -------------------------
def get_status(y_bottom, y_top):
    if y_bottom >= line_y_avg - 3 and y_top < line_y_avg:
        return "touch"
    elif y_top >= line_y_avg:
        return "cross"
    return None

DIST_THRESHOLD = 50

# -------------------------
# Video writer
# -------------------------
output_path = os.path.join(OUTPUT_DIR, "processed_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -------------------------
# Video Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    traffic_results = traffic_model(frame, verbose=False)[0]
    bike_boxes = []

    # -------------------- Traffic Detection --------------------
    for box in traffic_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = traffic_model.names[cls_id].lower()
        if label == "person":
            continue

        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        overlap_line = not (x2 < line_x_min or x1 > line_x_max)

        color = (0, 255, 0)
        violation = False

        if overlap_line:
            status = get_status(y2, y1)
            if status == "touch":
                color = (0, 0, 255)
            elif status == "cross":
                color = (0, 255, 0)
                violation = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if violation:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(TRAFFIC_DIR, f"{label}_{timestamp}.jpg")
            cv2.imwrite(save_path, frame[y1:y2, x1:x2])

        if label in ["motorbike", "bicycle"]:
            bike_boxes.append((x1, y1, x2, y2, label))

    # -------------------- Helmet Detection --------------------
    if bike_boxes:
        helmet_results = helmet_model(frame, verbose=False)[0]
        helmet_coords = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in helmet_results.boxes.xyxy]

        for (x1, y1, x2, y2, label) in bike_boxes:
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            has_helmet = any(hx1 <= cx <= hx2 and hy1 <= cy <= hy2 for hx1, hy1, hx2, hy2 in helmet_coords)

            color = (0, 255, 0) if has_helmet else (0, 0, 255)
            text = f"{label} - {'Helmet' if has_helmet else 'No Helmet'}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not has_helmet:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(HELMET_DIR, f"{label}_{timestamp}.jpg")
                cv2.imwrite(save_path, frame[y1:y2, x1:x2])

    # -------------------- Draw Line --------------------
    cv2.line(frame, line_points[0], line_points[1], (255, 255, 0), 2)
    cv2.putText(frame, "Boundary Line", (line_points[0][0]+10, line_points[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Write frame to output video
    out.write(frame)
    cv2.imshow("Traffic & Helmet Violation Detection", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Detection completed! Output video saved at:\n{output_path}")


