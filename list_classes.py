# scripts/list_classes.py
import os
import xml.etree.ElementTree as ET
from collections import Counter

ann_dir = r"D:\TrafficViolationDetection\helmet_detection\annotations"  # adjust path
names = Counter()

for fname in os.listdir(ann_dir):
    if not fname.endswith(".xml"):
        continue
    path = os.path.join(ann_dir, fname)
    tree = ET.parse(path)
    for obj in tree.findall("object"):
        name = obj.find("name").text.strip()
        names[name] += 1

print("Found classes and counts:")
for k,v in names.items():
    print(f"  {k}: {v}")
