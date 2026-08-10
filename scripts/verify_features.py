import os
import glob
import json
import numpy as np
from collections import Counter

FEATURE_FOLDER = "features"

print("\n============================================================")
print("EXP-02 FEATURE VERIFICATION")
print("============================================================")


# ============================================================
# LOAD CONFIG
# ============================================================

config_path = os.path.join(
    FEATURE_FOLDER,
    "experiment_config.json"
)

with open(config_path, "r") as file:
    config = json.load(file)

print("\nExperiment:", config["experiment"])
print("Backbone:", config["backbone"])
print("Sequence length:", config["sequence_length"])
print("Feature size:", config["feature_size"])
print("Expected classes:", len(config["classes"]))


# ============================================================
# FIND CHUNKS
# ============================================================

chunk_files = sorted(
    glob.glob(
        os.path.join(
            FEATURE_FOLDER,
            "chunk_*.npz"
        )
    )
)

print("\nChunk files found:", len(chunk_files))


# ============================================================
# VERIFY DATA
# ============================================================

total_samples = 0

all_labels = []
all_label_names = []

nan_count = 0
inf_count = 0

zero_feature_samples = 0

bad_shape_chunks = []

feature_min = float("inf")
feature_max = float("-inf")
feature_sum = 0.0
feature_value_count = 0


for chunk_number, chunk_file in enumerate(
    chunk_files,
    start=1
):

    data = np.load(chunk_file)

    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    video_paths = data["video_paths"]


    expected_shape = (
        len(features),
        config["sequence_length"],
        config["feature_size"]
    )


    if features.shape != expected_shape:

        bad_shape_chunks.append(
            (
                chunk_file,
                features.shape
            )
        )


    if not (
        len(features)
        == len(labels)
        == len(label_names)
        == len(video_paths)
    ):

        print(
            "\nLENGTH MISMATCH:",
            chunk_file
        )


    total_samples += len(features)

    all_labels.extend(
        labels.tolist()
    )

    all_label_names.extend(
        label_names.tolist()
    )


    nan_count += int(
        np.isnan(features).sum()
    )

    inf_count += int(
        np.isinf(features).sum()
    )


    flattened_per_sample = features.reshape(
        len(features),
        -1
    )

    zero_feature_samples += int(
        np.all(
            flattened_per_sample == 0,
            axis=1
        ).sum()
    )


    feature_min = min(
        feature_min,
        float(features.min())
    )

    feature_max = max(
        feature_max,
        float(features.max())
    )

    feature_sum += float(
        features.sum(
            dtype=np.float64
        )
    )

    feature_value_count += features.size


    if (
        chunk_number == 1
        or chunk_number % 10 == 0
        or chunk_number == len(chunk_files)
    ):

        print(
            f"Checked "
            f"{chunk_number}/{len(chunk_files)} "
            f"chunks"
        )


# ============================================================
# SUMMARY
# ============================================================

feature_mean = (
    feature_sum / feature_value_count
)

label_counts = Counter(
    all_label_names
)

unique_numeric_labels = sorted(
    set(all_labels)
)


print("\n============================================================")
print("VERIFICATION SUMMARY")
print("============================================================")

print("\nTotal chunks:", len(chunk_files))
print("Total samples:", total_samples)

print(
    "Unique numeric labels:",
    unique_numeric_labels
)

print(
    "Number of classes:",
    len(label_counts)
)


print("\nCLASS DISTRIBUTION")

for class_name in config["classes"]:

    print(
        f"{class_name:20} "
        f"{label_counts[class_name]:5}"
    )


print("\nFEATURE HEALTH")

print("NaN values:", nan_count)
print("Infinity values:", inf_count)

print(
    "All-zero feature samples:",
    zero_feature_samples
)

print(
    "Bad shape chunks:",
    len(bad_shape_chunks)
)

print(
    "Feature minimum:",
    feature_min
)

print(
    "Feature maximum:",
    feature_max
)

print(
    "Feature mean:",
    feature_mean
)


# ============================================================
# FINAL STATUS
# ============================================================

expected_classes = len(
    config["classes"]
)

healthy = True


if total_samples != 10091:

    print(
        "\nWARNING: Expected 10091 samples."
    )

    healthy = False


if len(chunk_files) != 101:

    print(
        "\nWARNING: Expected 101 chunks."
    )

    healthy = False


if len(label_counts) != expected_classes:

    print(
        "\nWARNING: Class count mismatch."
    )

    healthy = False


if nan_count > 0:

    print(
        "\nWARNING: NaN values detected."
    )

    healthy = False


if inf_count > 0:

    print(
        "\nWARNING: Infinity values detected."
    )

    healthy = False


if zero_feature_samples > 0:

    print(
        "\nWARNING: All-zero samples detected."
    )

    healthy = False


if bad_shape_chunks:

    print(
        "\nBAD SHAPE DETAILS"
    )

    for item in bad_shape_chunks:

        print(item)

    healthy = False


print("\n============================================================")

if healthy:

    print(
        "STATUS: FEATURE DATA IS HEALTHY AND READY FOR GRU"
    )

else:

    print(
        "STATUS: FEATURE DATA NEEDS INVESTIGATION"
    )

print("============================================================")
