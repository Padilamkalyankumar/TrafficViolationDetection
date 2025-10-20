# scripts/split_dataset.py
import os, shutil, random
from glob import glob

random.seed(42)

SRC_IMG_DIR = r"D:\TrafficViolationDetection\helmet_detection\images"      # adjust
SRC_LABEL_DIR = r"D:\TrafficViolationDetection\helmet_detection\labels"    # output from previous step

OUT_ROOT = r"D:\TrafficViolationDetection\models\helmet_dataset"  # target folder
for split in ["train","val","test"]:
    os.makedirs(os.path.join(OUT_ROOT, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, split, "labels"), exist_ok=True)

# collect images (jpg/png) that have a matching label file
img_paths = []
for ext in ("*.jpg","*.jpeg","*.png"):
    img_paths += glob(os.path.join(SRC_IMG_DIR, ext))

paired = []
for img in img_paths:
    base = os.path.splitext(os.path.basename(img))[0]
    label = os.path.join(SRC_LABEL_DIR, base + ".txt")
    if os.path.exists(label):
        paired.append((img, label))
    else:
        # optionally ignore images without labels
        pass

random.shuffle(paired)
n = len(paired)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train = paired[:n_train]
val = paired[n_train:n_train+n_val]
test = paired[n_train+n_val:]

def copy_pairs(pairs, split):
    for img, lab in pairs:
        shutil.copy(img, os.path.join(OUT_ROOT, split, "images", os.path.basename(img)))
        shutil.copy(lab, os.path.join(OUT_ROOT, split, "labels", os.path.basename(lab)))

copy_pairs(train, "train")
copy_pairs(val, "val")
copy_pairs(test, "test")
print(f"Copied {len(train)} train, {len(val)} val, {len(test)} test samples to {OUT_ROOT}")
