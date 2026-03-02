# =====================================================
# HELMET DETECTION USING YOLOv8
# FINAL OUTPUT → helmet_model.pt
# =====================================================

# ------------------------------
# 1. INSTALL
# ------------------------------
# !pip install -q ultralytics

# Disable wandb logging
import os
os.environ["WANDB_DISABLED"] = "true"

# ------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------
import shutil
import yaml
import glob
from xml.dom import minidom
from ultralytics import YOLO

print("YOLOv8 Ready ✅")

# ------------------------------
# 3. DATASET YAML
# ------------------------------
data_config = {
    "train": "/kaggle/working/images",
    "val": "/kaggle/working/images",
    "nc": 2,
    "names": ["Without Helmet", "With Helmet"]
}

with open("dataset.yaml", "w") as f:
    yaml.dump(data_config, f)

# ------------------------------
# 4. COPY DATASET
# ------------------------------
# !cp -r /kaggle/input/helmet-detection/images/ /kaggle/working/

label_dir = "/kaggle/working/labels"

if os.path.exists(label_dir):
    shutil.rmtree(label_dir)

os.makedirs(label_dir)

print("Dataset copied ✅")

# ------------------------------
# 5. XML → YOLO LABEL CONVERSION
# ------------------------------
class_map = {"Without Helmet": 0, "With Helmet": 1}

def read_tag(tag, parent):
    return parent.getElementsByTagName(tag)[0].firstChild.data

def convert_box(size, box):
    W, H = size
    x_center = (box[0] + box[1]) / 2 / W
    y_center = (box[2] + box[3]) / 2 / H
    width = (box[1] - box[0]) / W
    height = (box[3] - box[2]) / H
    return x_center, y_center, width, height

xml_folder = "/kaggle/input/helmet-detection/annotations"

for xml in glob.glob(xml_folder + "/*.xml"):

    doc = minidom.parse(xml)

    size_tag = doc.getElementsByTagName("size")[0]
    W = int(read_tag("width", size_tag))
    H = int(read_tag("height", size_tag))

    out_file = os.path.join(
        label_dir,
        os.path.basename(xml).replace(".xml", ".txt")
    )

    with open(out_file, "w") as f:
        for obj in doc.getElementsByTagName("object"):

            cls = read_tag("name", obj)
            if cls not in class_map:
                continue

            bb = obj.getElementsByTagName("bndbox")[0]

            coords = [
                float(read_tag("xmin", bb)),
                float(read_tag("xmax", bb)),
                float(read_tag("ymin", bb)),
                float(read_tag("ymax", bb))
            ]

            x, y, w, h = convert_box((W, H), coords)

            f.write(f"{class_map[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("Labels converted ✅")

# ------------------------------
# 6. TRAIN YOLO MODEL
# ------------------------------
model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=16
)

print("Training Finished ✅")

# ------------------------------
# 7. SAVE ONLY FINAL MODEL (.pt)
# ------------------------------
best_model = "/kaggle/working/runs/detect/train/weights/best.pt"

shutil.copy(best_model,
            "/kaggle/working/helmet_model.pt")

print("\n🎯 helmet_model.pt SAVED")
print("Download from Kaggle → Output")