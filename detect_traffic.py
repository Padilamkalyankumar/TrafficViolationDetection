import cv2
import os
import shutil
from datetime import datetime
from ultralytics import YOLO

# -----------------------------
# Configuration
# -----------------------------
video_path = r"D:\TrafficViolationDetection\videos\sample1.mp4.f398.mp4"
model_path = r"D:\TrafficViolationDetection\src\yolo11n.pt"
save_dir = r"D:\TrafficViolationDetection\detected_violations"
os.makedirs(save_dir, exist_ok=True)

# Delete old files and folders
for item in os.listdir(save_dir):
    item_path = os.path.join(save_dir, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

# Load model
model = YOLO(model_path)

# Vehicle classes to track
vehicle_classes = ["car", "bus", "truck", "motorcycle", "bicycle", "auto", "autorickshaw"]

# -----------------------------
# User draws the violation line
# -----------------------------
line_pts = []

def draw_line(event, x, y, flags, param):
    global line_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        line_pts.append((x, y))
        if len(line_pts) == 2:
            cv2.line(frame_copy, line_pts[0], line_pts[1], (0, 255, 255), 2)
            cv2.imshow("Draw Line - Click 2 points", frame_copy)

cap = cv2.VideoCapture(video_path)
ret, frame_copy = cap.read()
if not ret:
    print("Error: Could not read video")
    cap.release()
    exit()

cv2.imshow("Draw Line - Click 2 points", frame_copy)
cv2.setMouseCallback("Draw Line - Click 2 points", draw_line)

print("Please click 2 points to define the violation line.")
while len(line_pts) < 2:
    cv2.waitKey(1)

cv2.destroyAllWindows()

# Line coordinates
x1_line, y1_line = line_pts[0]
x2_line, y2_line = line_pts[1]

# Horizontal boundaries for lane width
line_x_min, line_x_max = min(x1_line, x2_line), max(x1_line, x2_line)
line_y_avg = (y1_line + y2_line) // 2  # average y for touch/cross detection

# -----------------------------
# Helper function
# -----------------------------
def get_status(y_bottom, y_top):
    """
    Returns:
        'touch' -> if vehicle bottom touches the line
        'cross' -> if fully crossed the line
        None -> otherwise
    """
    if y_bottom >= line_y_avg - 3 and y_top < line_y_avg:
        return "touch"
    elif y_top >= line_y_avg:
        return "cross"
    return None

# -----------------------------
# Main Loop
# -----------------------------
frame_count = 0
saved_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % 2 != 0:  # process alternate frames for speed
        continue

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()

        # Skip non-vehicle classes
        if cls_name not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        y_bottom, y_top = y2, y1

        # Check if vehicle overlaps line horizontally
        overlap_line = not (x2 < line_x_min or x1 > line_x_max)

        color = (0, 255, 0)  # default green

        if overlap_line:
            status = get_status(y_bottom, y_top)
            if status == "touch":
                color = (0, 0, 255)  # red
            elif status == "cross":
                color = (0, 255, 0)
                vehicle_id = f"{cls_name}_{x1}_{y1}"
                if vehicle_id not in saved_ids:
                    crop = frame[y1:y2, x1:x2]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(save_dir, f"{cls_name}_{timestamp}.jpg")
                    cv2.imwrite(filename, crop)
                    saved_ids.add(vehicle_id)
        # Vehicles outside line width: green box only

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls_name, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw violation line
    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (255, 255, 0), 3)
    cv2.putText(frame, "Violation Line", (x1_line, y1_line - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
