import os
import cv2
import numpy as np
import subprocess

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 300
NUM_FRAMES = 30


# ============================================================
# AI VISION
# ============================================================

def generate_ai_vision(video_path, output_dir):

    frames_dir = os.path.join(output_dir, "frames")
    overlays_dir = os.path.join(output_dir, "overlays")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    print("=" * 60)
    print("Loading EfficientNetB3...")
    print("=" * 60)

    model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    print("Model loaded successfully.\n")

    # ========================================================
    # READ VIDEO
    # ========================================================

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No readable frames found.")

    # ========================================================
    # SAMPLE FRAMES
    # ========================================================

    if len(frames) >= NUM_FRAMES:

        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES,
            dtype=int
        )

        sampled = [frames[i] for i in indices]

    else:

        sampled = frames.copy()

        while len(sampled) < NUM_FRAMES:
            sampled.append(sampled[-1])

    # ========================================================
    # CREATE OVERLAYS
    # ========================================================

    print("=" * 60)
    print("Generating AI Vision overlays...")
    print("=" * 60)

    for i, frame in enumerate(sampled):

        original = frame.copy()

        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        rgb = rgb.astype(np.float32)

        x = preprocess_input(rgb)
        x = np.expand_dims(x, axis=0)

        feature_maps = model.predict(
            x,
            verbose=0
        )[0]

        activation = np.mean(
            feature_maps,
            axis=-1
        )

        activation -= activation.min()

        if activation.max() > 0:
            activation /= activation.max()

        activation = cv2.resize(
            activation,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        heatmap = np.uint8(255 * activation)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        filename = os.path.join(
            overlays_dir,
            f"{i:02d}.jpg"
        )

        cv2.imwrite(filename, overlay)

        print(f"[{i+1:02d}/{NUM_FRAMES}] Done")

    print("\nFinished!")
    print(f"Saved overlays to:\n{overlays_dir}")

    # ========================================================
    # CREATE VIDEO
    # ========================================================

    print("\n" + "=" * 60)
    print("Creating AI Vision Video...")
    print("=" * 60)

    output_video_path = os.path.join(
        output_dir,
        "AI_Vision.mp4"
    )

    first_frame = cv2.imread(
        os.path.join(
            overlays_dir,
            "00.jpg"
        )
    )

    height, width = first_frame.shape[:2]

    temp_video = os.path.join(
        output_dir,
        "temp.mp4"
    )

    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (width, height)
    )

    for i in range(NUM_FRAMES):

        frame = cv2.imread(
            os.path.join(
                overlays_dir,
                f"{i:02d}.jpg"
            )
        )

        writer.write(frame)

    writer.release()

    print("\nConverting to H.264...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp_video,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_video_path,
        ],
        check=True,
    )

    os.remove(temp_video)

    print("\nVideo saved successfully!")
    print(output_video_path)

    return output_video_path