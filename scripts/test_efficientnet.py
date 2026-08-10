import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

SEQUENCE_LENGTH = 30
IMAGE_SIZE = 224

TEST_FOLDER = "dataset/Cover Drive"
NUM_TEST_VIDEOS = 5

print("\n===== EFFICIENTNET MINI TEST =====")

print("\nTensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

print("\nLoading EfficientNetB0...")

model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("EfficientNet loaded!")

videos = sorted([
    file
    for file in os.listdir(TEST_FOLDER)
    if file.lower().endswith(".avi")
])[:NUM_TEST_VIDEOS]

print(f"\nTesting {len(videos)} videos from: {TEST_FOLDER}")

all_video_features = []

for video_number, video_file in enumerate(videos, start=1):

    video_path = os.path.join(TEST_FOLDER, video_file)

    print(
        f"\n[{video_number}/{len(videos)}] "
        f"Reading: {video_file}"
    )

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < 1:
        cap.release()
        print("Could not read video.")
        continue

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
            frame_index
        )

        success, frame = cap.read()

        if not success:
            continue

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

    if len(frames) != SEQUENCE_LENGTH:
        print(
            f"Skipped: only {len(frames)} "
            f"of {SEQUENCE_LENGTH} frames read."
        )
        continue

    frames = np.array(
        frames,
        dtype=np.float32
    )

    frames = preprocess_input(frames)

    print("Frames shape:", frames.shape)

    features = model.predict(
        frames,
        batch_size=10,
        verbose=0
    )

    print("EfficientNet output:", features.shape)

    all_video_features.append(features)

if all_video_features:

    all_video_features = np.array(
        all_video_features
    )

    print("\n===== FINAL RESULT =====")
    print(
        "All video features shape:",
        all_video_features.shape
    )

    print("\nSUCCESS: EfficientNet pipeline works!")

else:

    print("\nFAILED: No videos were processed.")

print("\n===== TEST COMPLETE =====")
