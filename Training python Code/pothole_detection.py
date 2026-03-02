# =====================================================
# POTHOLE DETECTION (BOUNDING BOX REGRESSION)
# FINAL OUTPUT → pothole_model.pt
# =====================================================

# ------------------------------
# 1. INSTALL LIBRARIES
# ------------------------------
# !pip install albumentations timm --quiet

# ------------------------------
# 2. IMPORTS
# ------------------------------
import os
import cv2
import torch
import numpy as np
import pandas as pd
from xml.dom import minidom
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
import torch.nn as nn
import timm

# ------------------------------
# 3. SETTINGS
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

DATA_PATH = "/kaggle/input/annotated-potholes-dataset/annotated-images"
MODEL_NAME = "res2net50d.in1k"

print("Using device:", DEVICE)

# ------------------------------
# 4. READ XML BOX
# ------------------------------
def extract_box(xml_file):
    data = minidom.parse(xml_file)

    xmin = int(data.getElementsByTagName("xmin")[0].childNodes[0].data)
    ymin = int(data.getElementsByTagName("ymin")[0].childNodes[0].data)
    xmax = int(data.getElementsByTagName("xmax")[0].childNodes[0].data)
    ymax = int(data.getElementsByTagName("ymax")[0].childNodes[0].data)

    return xmin, ymin, xmax, ymax

# ------------------------------
# 5. BUILD DATAFRAME
# ------------------------------
def build_csv(path):

    records = []

    for f in os.listdir(path):

        if f.endswith(".jpg"):

            img_path = os.path.join(path, f)
            xml_path = img_path.replace(".jpg", ".xml")

            if os.path.exists(xml_path):

                xmin, ymin, xmax, ymax = extract_box(xml_path)

                img_arr = cv2.imread(img_path)
                h, w = img_arr.shape[:2]

                records.append([img_path, w, h, xmin, ymin, xmax, ymax])

    df = pd.DataFrame(
        records,
        columns=["file","w","h","xmin","ymin","xmax","ymax"]
    )

    return df


df = build_csv(DATA_PATH)

print("Total samples:", len(df))

if len(df) == 0:
    raise ValueError("Dataset empty — path incorrect.")

# ------------------------------
# 6. TRAIN / VAL SPLIT
# ------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42
)

# ------------------------------
# 7. AUGMENTATIONS
# ------------------------------
train_tf = A.Compose(
    [A.Resize(IMG_SIZE, IMG_SIZE)],
    bbox_params=A.BboxParams(format="pascal_voc",
                             label_fields=["labels"])
)

val_tf = A.Compose(
    [A.Resize(IMG_SIZE, IMG_SIZE)],
    bbox_params=A.BboxParams(format="pascal_voc",
                             label_fields=["labels"])
)

# ------------------------------
# 8. DATASET CLASS
# ------------------------------
class PotholeDS(torch.utils.data.Dataset):

    def __init__(self, df, aug):
        self.df = df.reset_index(drop=True)
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        r = self.df.iloc[idx]

        img = cv2.imread(r.file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox = [[r.xmin, r.ymin, r.xmax, r.ymax]]
        labels = [1]

        data = self.aug(image=img,
                        bboxes=bbox,
                        labels=labels)

        img = torch.tensor(data["image"]).permute(2,0,1).float()/255.0
        bbox = torch.tensor(data["bboxes"][0]).float()

        return img, bbox


train_loader = torch.utils.data.DataLoader(
    PotholeDS(train_df, train_tf),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    PotholeDS(val_df, val_tf),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------
# 9. MODEL
# ------------------------------
class BoxModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=4
        )

    def forward(self, x, y=None):

        pred = self.net(x)

        if y is not None:
            loss = nn.SmoothL1Loss()(pred.float(), y.float())
            return pred, loss

        return pred


model = BoxModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------------
# 10. TRAIN LOOP
# ------------------------------
best_loss = 1e9

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for img, bbox in tqdm(train_loader):

        img = img.to(DEVICE)
        bbox = bbox.to(DEVICE)

        pred, loss = model(img, bbox)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, bbox in val_loader:
            img = img.to(DEVICE)
            bbox = bbox.to(DEVICE)

            _, loss = model(img, bbox)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "pothole_model.pt")
        print("✅ Saved Best Model")

# ------------------------------
# 11. FINAL SAVE (ONLY FILE)
# ------------------------------
torch.save(model.state_dict(), "pothole_model.pt")

print("\n🎯 FINAL OUTPUT SAVED → pothole_model.pt")