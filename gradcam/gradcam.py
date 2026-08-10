import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "gradcam/output/frames/00.jpg"
OUTPUT_PATH = "gradcam/output/heatmap_00.jpg"

# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception("Could not load image.")

original = image.copy()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (300, 300))
rgb = rgb.astype(np.float32)

input_tensor = preprocess_input(rgb)
input_tensor = np.expand_dims(input_tensor, axis=0)

# ============================================================
# LOAD MODEL
# ============================================================

model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    pooling=None,
    input_shape=(300, 300, 3)
)

# ============================================================
# GET FEATURE MAPS
# ============================================================

feature_maps = model.predict(input_tensor, verbose=0)[0]

print("Feature map shape:", feature_maps.shape)

# ============================================================
# CREATE BETTER ACTIVATION MAP
# ============================================================

# Compute importance of each channel
channel_strength = np.mean(feature_maps, axis=(0, 1))

# Keep only the strongest 10% of channels
threshold = np.percentile(channel_strength, 90)

strong_channels = np.where(channel_strength >= threshold)[0]

# Average only those channels
activation = np.mean(feature_maps[:, :, strong_channels], axis=-1)

# Normalize
activation -= activation.min()

if activation.max() > 0:
    activation /= activation.max()

# Smooth the heatmap
activation = cv2.GaussianBlur(activation, (5, 5), 0)

# Suppress weak responses
activation[activation < 0.30] = 0