import os
import xml.etree.ElementTree as ET

# Paths
dataset_path = r"D:\TrafficViolationDetection\bikerider-dataset"
sets = ["train", "test"]

# Define your classes
classes = ["helmet"]   # Add more if you have other labels like "no-helmet"

def convert(size, box):
    """Convert VOC bbox to YOLO format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

for image_set in sets:
    img_path = os.path.join(dataset_path, "images", image_set)
    label_path = os.path.join(dataset_path, "labels", image_set)

    os.makedirs(label_path, exist_ok=True)

    for file in os.listdir(img_path):
        if not file.endswith(".xml"):
            continue

        xml_file = os.path.join(img_path, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_w = int(root.find("size/width").text)
        img_h = int(root.find("size/height").text)

        out_file = open(os.path.join(label_path, file.replace(".xml", ".txt")), "w")

        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((img_w, img_h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
        out_file.close()

print("✅ Conversion complete! YOLO labels saved inside 'labels/train' and 'labels/test'.")
