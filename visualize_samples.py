# scripts/visualize_samples.py
import os
import cv2
import random
import matplotlib.pyplot as plt

# Path to images folder
images_dir = r"D:\TrafficViolationDetection\helmet_detection\images"
annotations_dir = r"D:\TrafficViolationDetection\helmet_detection\annotations"

# Get random sample image
all_images = os.listdir(images_dir)
sample_img = random.choice(all_images)
img_path = os.path.join(images_dir, sample_img)

# Read image
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Could not load {img_path}")
    exit()

# Read annotation file (YOLO format assumed)
label_path = os.path.join(
    annotations_dir, os.path.splitext(sample_img)[0] + ".txt"
)
if os.path.exists(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    h, w, _ = img.shape
    for line in lines:
        cls, cx, cy, bw, bh = map(float, line.strip().split())
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert BGR → RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show with matplotlib
plt.imshow(img_rgb)
plt.title(f"Sample: {sample_img}")
plt.axis("off")
plt.show()
