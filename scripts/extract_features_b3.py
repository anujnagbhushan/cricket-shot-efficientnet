import os
import time
import json
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "EXP-05_EFFICIENTNETB3_GRU"

SEQUENCE_LENGTH = 30
IMAGE_SIZE = 300
FEATURE_SIZE = 1536

CHUNK_SIZE = 100
FRAME_BATCH_SIZE = 30

OUTPUT_FOLDER = "features_b3"


CLASS_FOLDERS = [
    "dataset/Cover Drive",
    "dataset/Defensive",
    "dataset/Down The Wicket",
    "dataset/Flick",
    "dataset/Hook",
    "dataset/Late Cut",
    "dataset/Lofted Legside",
    "dataset/Lofted Offside",
    "dataset/Pull",
    "dataset/Reverse Sweep",
    "dataset/Scoop",
    "dataset/Square Cut",
    "dataset/Straight Drive",
    "dataset/Sweep",
    "dataset/Upper Cut"
]


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\n============================================================")
print("EXP-02 FULL FEATURE EXTRACTION")
print("============================================================")

print("\nExperiment:", EXPERIMENT_NAME)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Sequence length:", SEQUENCE_LENGTH)
print("Image size:", IMAGE_SIZE)
print("Feature size:", FEATURE_SIZE)
print("Chunk size:", CHUNK_SIZE)


# ============================================================
# SAVE EXPERIMENT CONFIG
# ============================================================

config = {
    "experiment": EXPERIMENT_NAME,
    "backbone": "EfficientNetB3",
    "weights": "ImageNet",
    "include_top": False,
    "pooling": "avg",
    "sequence_length": SEQUENCE_LENGTH,
    "image_size": IMAGE_SIZE,
    "feature_size": FEATURE_SIZE,
    "chunk_size": CHUNK_SIZE,
    "classes": CLASS_FOLDERS
}

config_path = os.path.join(
    OUTPUT_FOLDER,
    "experiment_config.json"
)

with open(config_path, "w") as file:
    json.dump(config, file, indent=4)

print("\nExperiment config saved:", config_path)


# ============================================================
# BUILD VIDEO LIST
# ============================================================

video_records = []

for label_index, folder in enumerate(CLASS_FOLDERS):

    videos = sorted([
        file
        for file in os.listdir(folder)
        if file.lower().endswith(".avi")
    ])

    for video_file in videos:

        video_records.append({
            "path": os.path.join(folder, video_file),
            "label_index": label_index,
            "label_name": folder
        })


total_videos = len(video_records)

print("\nTotal videos found:", total_videos)
print("Total classes:", len(CLASS_FOLDERS))


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

print("\nLoading EfficientNetB3...")

model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False

print("EfficientNet loaded!")
print("Expected output per frame:", FEATURE_SIZE)


# ============================================================
# VIDEO PROCESSING FUNCTION
# ============================================================

def extract_video_features(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < 1:
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            cap.release()
            return None

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

    frames = np.asarray(
        frames,
        dtype=np.float32
    )

    frames = preprocess_input(frames)

    features = model.predict(
        frames,
        batch_size=FRAME_BATCH_SIZE,
        verbose=0
    )

    if features.shape != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):
        return None

    return features.astype(np.float32)


# ============================================================
# EXTRACTION
# ============================================================

total_chunks = (
    total_videos + CHUNK_SIZE - 1
) // CHUNK_SIZE

global_start_time = time.time()

total_processed_this_run = 0
total_skipped_videos = 0


for chunk_index in range(total_chunks):

    chunk_filename = (
        f"chunk_{chunk_index:04d}.npz"
    )

    chunk_path = os.path.join(
        OUTPUT_FOLDER,
        chunk_filename
    )

    if os.path.exists(chunk_path):

        print(
            f"\n[CHUNK {chunk_index + 1}/{total_chunks}] "
            f"{chunk_filename} already exists -> SKIP"
        )

        continue


    start_index = chunk_index * CHUNK_SIZE

    end_index = min(
        start_index + CHUNK_SIZE,
        total_videos
    )

    chunk_records = video_records[
        start_index:end_index
    ]


    print("\n============================================================")
    print(
        f"CHUNK {chunk_index + 1}/{total_chunks}"
    )
    print(
        f"Videos {start_index + 1} to {end_index}"
    )
    print("============================================================")


    chunk_features = []
    chunk_labels = []
    chunk_label_names = []
    chunk_video_paths = []

    chunk_start_time = time.time()


    for local_index, record in enumerate(
        chunk_records,
        start=1
    ):

        video_number = (
            start_index + local_index
        )

        video_path = record["path"]

        features = extract_video_features(
            video_path
        )

        if features is None:

            total_skipped_videos += 1

            print(
                f"[{video_number}/{total_videos}] "
                f"SKIPPED: {video_path}"
            )

            continue


        chunk_features.append(features)

        chunk_labels.append(
            record["label_index"]
        )

        chunk_label_names.append(
            record["label_name"]
        )

        chunk_video_paths.append(
            video_path
        )

        total_processed_this_run += 1


        elapsed = (
            time.time() - global_start_time
        )

        speed = (
            total_processed_this_run / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total_videos - video_number
        )

        eta_seconds = (
            remaining / speed
            if speed > 0
            else 0
        )

        eta_hours = eta_seconds / 3600


        print(
            f"[{video_number}/{total_videos}] "
            f"{record['label_name']} | "
            f"Speed: {speed:.2f} videos/sec | "
            f"ETA: {eta_hours:.2f} hours"
        )


    if len(chunk_features) == 0:

        print(
            "\nWARNING: No valid videos in this chunk."
        )

        continue


    chunk_features = np.asarray(
        chunk_features,
        dtype=np.float32
    )

    chunk_labels = np.asarray(
        chunk_labels,
        dtype=np.int32
    )

    chunk_label_names = np.asarray(
        chunk_label_names
    )

    chunk_video_paths = np.asarray(
        chunk_video_paths
    )


    np.savez(
        chunk_path,
        features=chunk_features,
        labels=chunk_labels,
        label_names=chunk_label_names,
        video_paths=chunk_video_paths
    )


    chunk_elapsed = (
        time.time() - chunk_start_time
    )

    print("\nCHUNK SAVED:", chunk_path)
    print(
        "Feature shape:",
        chunk_features.shape
    )
    print(
        f"Chunk time: "
        f"{chunk_elapsed / 60:.2f} minutes"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_elapsed = (
    time.time() - global_start_time
)

print("\n============================================================")
print("EXTRACTION COMPLETE")
print("============================================================")

print(
    "Processed this run:",
    total_processed_this_run
)

print(
    "Skipped videos:",
    total_skipped_videos
)

print(
    f"Total runtime: "
    f"{total_elapsed / 3600:.2f} hours"
)

print(
    "Features saved in:",
    OUTPUT_FOLDER
)

print("============================================================")
