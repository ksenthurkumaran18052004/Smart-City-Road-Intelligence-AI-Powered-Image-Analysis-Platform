# =====================================================
# LICENSE PLATE DETECTION - YOLOv8 CLEAN PIPELINE
# =====================================================

!pip install -q ultralytics

import os
import shutil
import yaml
import glob
import random
import cv2
import torch
import pytesseract
import numpy as np
from xml.dom import minidom
from ultralytics import YOLO

# ------------------------------
# 1. CLASS LOOKUP TABLE
# ------------------------------
lut = {"number_plate": 0}

# ------------------------------
# 2. XML → YOLO CONVERSION
# ------------------------------
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x*dw, y*dh, w*dw, h*dh)

def convert_xml2yolo(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for fname in glob.glob(f"{input_path}/*.xml"):
        xmldoc = minidom.parse(fname)
        size = xmldoc.getElementsByTagName('size')[0]

        width = int(size.getElementsByTagName('width')[0].firstChild.data)
        height = int(size.getElementsByTagName('height')[0].firstChild.data)

        output_file = os.path.join(
            output_path,
            os.path.basename(fname).replace(".xml", ".txt")
        )

        with open(output_file, "w") as f:
            for obj in xmldoc.getElementsByTagName('object'):

                xmin = float(obj.getElementsByTagName('xmin')[0].firstChild.data)
                ymin = float(obj.getElementsByTagName('ymin')[0].firstChild.data)
                xmax = float(obj.getElementsByTagName('xmax')[0].firstChild.data)
                ymax = float(obj.getElementsByTagName('ymax')[0].firstChild.data)

                bb = convert_coordinates((width, height),
                                         (xmin, xmax, ymin, ymax))

                f.write(f"0 {' '.join([f'{x:.6f}' for x in bb])}\n")

    print("✅ XML conversion done")

convert_xml2yolo(
    "/kaggle/input/car-plate-detection/annotations",
    "/kaggle/working/labels"
)

# ------------------------------
# 3. CREATE YOLO DATASET STRUCTURE
# ------------------------------
base_dir = "/kaggle/working/plate_dataset"

dirs = [
    f"{base_dir}/images/train",
    f"{base_dir}/images/val",
    f"{base_dir}/labels/train",
    f"{base_dir}/labels/val",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# ------------------------------
# 4. SPLIT TRAIN / VAL
# ------------------------------
images = os.listdir("/kaggle/input/car-plate-detection/images")
split = int(len(images) * 0.9)

for i, img in enumerate(images):
    subset = "train" if i < split else "val"

    os.system(
        f"cp /kaggle/input/car-plate-detection/images/{img} "
        f"{base_dir}/images/{subset}/{img}"
    )

    label_name = img.split(".")[0] + ".txt"
    os.system(
        f"cp /kaggle/working/labels/{label_name} "
        f"{base_dir}/labels/{subset}/{label_name}"
    )

print("✅ Train/Val split complete")

# ------------------------------
# 5. CREATE DATA YAML
# ------------------------------
data_yaml = {
    "train": f"{base_dir}/images/train",
    "val": f"{base_dir}/images/val",
    "nc": 1,
    "names": ["number_plate"]
}

with open("/kaggle/working/plate_data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("✅ data.yaml created")

# ------------------------------
# 6. TRAIN YOLO MODEL
# ------------------------------
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data="/kaggle/working/plate_data.yaml",
    epochs=15,
    imgsz=640,
    batch=16,
    device=device,
    patience=15,
    augment=True
)

print("✅ Training completed")

# ------------------------------
# 7. SAVE MODEL
# ------------------------------
best_model_path = "/kaggle/working/runs/detect/train/weights/best.pt"
shutil.copy(best_model_path, "/kaggle/working/plate_model.pt")

print("🎯 plate_model.pt ready for download")