import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from gradcam.ai_vision import generate_ai_vision
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.logger import log


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 30
IMAGE_SIZE = 300
FEATURE_SIZE = 1536

MODEL_PATH = "models/exp05_efficientnetb3_gru.keras"


CLASS_NAMES = [
    "Cover Drive",
    "Defensive",
    "Down The Wicket",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted Legside",
    "Lofted Offside",
    "Pull",
    "Reverse Sweep",
    "Scoop",
    "Square Cut",
    "Straight Drive",
    "Sweep",
    "Upper Cut"
]
# ============================================================
# LOAD EFFICIENTNETB3
# ============================================================

print("\nLoading EfficientNetB3 feature extractor...")

feature_extractor = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

feature_extractor.trainable = False

print("✓ EfficientNetB3 loaded.")


# ============================================================
# LOAD TRAINED GRU MODEL
# ============================================================

print("\nLoading trained GRU model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

gru_model = load_model(MODEL_PATH)

print("✓ GRU model loaded.")

print("\nModel Input Shape :", gru_model.input_shape)
print("Model Output Shape:", gru_model.output_shape)
# ============================================================
# FEATURE EXTRACTION FROM VIDEO
# ============================================================
def extract_video_features(video_path):

    log("Opening video...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video:\n{video_path}")

    frames = []

    log("Reading video sequentially...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        frames.append(frame)

    cap.release()

    total_frames = len(frames)

    log(f"Readable Frames : {total_frames}")

    if total_frames == 0:
        raise ValueError("No readable frames found in video.")

    if total_frames >= SEQUENCE_LENGTH:

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )

        sampled_frames = [
            frames[i]
            for i in frame_indices
        ]

    else:

        log(f"Video has only {total_frames} frames. Padding to {SEQUENCE_LENGTH}.")

        sampled_frames = frames.copy()

        while len(sampled_frames) < SEQUENCE_LENGTH:
            sampled_frames.append(sampled_frames[-1].copy())

    sampled_frames = np.asarray(
        sampled_frames,
        dtype=np.float32
    )

    sampled_frames = preprocess_input(sampled_frames)

    log("Extracting EfficientNetB3 features...")

    features = feature_extractor.predict(
        sampled_frames,
        batch_size=30,
        verbose=0
    )

    log("Feature extraction complete.", "success")

    return features
# ============================================================
# PREDICT
# ============================================================
def predict(video_path):

    import time

    start_time = time.time()

    features = extract_video_features(video_path)

    features = np.expand_dims(features, axis=0)

    predictions = gru_model.predict(
        features,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    inference_time = time.time() - start_time

    log("Prediction complete.", "success")
    log(f"Prediction : {CLASS_NAMES[predicted_index]}", "success")
    log(f"Confidence : {predictions[predicted_index] * 100:.2f}%", "success")
    log(f"Inference Time : {inference_time:.2f} sec", "success")

    log("Generating AI Vision...")

    ai_vision_path = generate_ai_vision(
        video_path,
        "app/static/ai_vision"
    )

    log("AI Vision video generated.", "success")

    top5_indices = np.argsort(predictions)[::-1][:5]

    top5 = []

    for idx in top5_indices:

        top5.append({
            "shot": CLASS_NAMES[idx],
            "confidence": float(predictions[idx] * 100)
        })

    result = {

        "prediction": CLASS_NAMES[predicted_index],

        "confidence": float(
            predictions[predicted_index] * 100
        ),

        "top5": top5,

        "inference_time": inference_time,

        "ai_vision_video": ai_vision_path

    }

    return result
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python inference/predict_video.py <video>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print("Video not found.")

        sys.exit(1)

    result = predict(video_path)

    print("\n========================================")
    print("CRICKET SHOT CLASSIFIER")
    print("========================================")

    print("\nPrediction")

    print(
        f"🏏 {result['prediction']}"
    )

    print(
        f"\nConfidence : "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"\nInference Time : "
        f"{result['inference_time']:.2f} sec"
    )

    print("\nTop Predictions\n")

    for i, item in enumerate(result["top5"], start=1):

        print(
            f"{i}. "
            f"{item['shot']:<20}"
            f"{item['confidence']:.2f}%"
        )
        
