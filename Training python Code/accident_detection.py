# =====================================================
# ACCIDENT DETECTION (TensorFlow → PyTorch .PT EXPORT)
# ONLY OUTPUT: model.pt
# =====================================================

# ------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------
# !pip install -q tf2onnx onnx onnx2pytorch

# ------------------------------
# 2. IMPORTS
# ------------------------------
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import torch
from onnx2pytorch import ConvertModel
import tf2onnx
# ------------------------------
# 3. PARAMETERS
# ------------------------------
batch_size = 100
img_height = 250
img_width = 250

train_path = "/kaggle/input/accident-detection-from-cctv-footage/data/train"
val_path   = "/kaggle/input/accident-detection-from-cctv-footage/data/val"

# ------------------------------
# 4. LOAD DATASET
# ------------------------------
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_path,
    seed=101,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = training_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# 5. BUILD MODEL (MobileNetV2)
# ------------------------------
img_shape = (img_height, img_width, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape,
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.Conv2D(32,3,activation="relu"),
    layers.Conv2D(64,3,activation="relu"),
    layers.Conv2D(128,3,activation="relu"),
    layers.Flatten(),
    layers.Dense(len(class_names),activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------
# 6. TRAIN MODEL
# ------------------------------
model.fit(
    training_ds,
    validation_data=validation_ds,
    epochs=30
)

print("Training complete ✅")

# ------------------------------
# 7. EXPORT TF MODEL → ONNX
# ------------------------------
onnx_path = "/kaggle/working/model.onnx"

spec = (tf.TensorSpec((None,img_height,img_width,3), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=onnx_path
)

print("Converted to ONNX ✅")

# ------------------------------
# 8. ONNX → PYTORCH
# ------------------------------
import onnx
onnx_model = onnx.load(onnx_path)

pytorch_model = ConvertModel(onnx_model)

# ------------------------------
# 9. SAVE ONLY .PT MODEL
# ------------------------------
torch.save(
    pytorch_model.state_dict(),
    "/kaggle/working/model.pt"
)

print("\n🎯 SUCCESS — model.pt SAVED")
print("Download from Kaggle → Output")