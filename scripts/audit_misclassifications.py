import os
import glob
import csv
import shutil
import numpy as np
import tensorflow as tf

from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FOLDER = "features"
SPLIT_FILE = "group_split.npz"
MODEL_FILE = "exp02_efficientnetb0_gru.keras"

AUDIT_FOLDER = "misclassification_audit"
AUDIT_CSV = "misclassification_audit.csv"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1280

BATCH_SIZE = 32
TOP_CONFUSION_PAIRS = 10
VIDEOS_PER_PAIR = 10


print("\n============================================================")
print("EXP-02 MISCLASSIFICATION VISUAL AUDIT")
print("============================================================")

print("\nTensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD FEATURE CHUNKS
# ============================================================

print("\nLoading feature chunks...")


chunk_files = sorted(
    glob.glob(
        os.path.join(
            FEATURE_FOLDER,
            "chunk_*.npz"
        )
    )
)


if not chunk_files:
    raise FileNotFoundError(
        "No feature chunks found."
    )


feature_parts = []
label_parts = []
label_name_parts = []
video_path_parts = []


for index, chunk_file in enumerate(
    chunk_files,
    start=1
):

    data = np.load(chunk_file)

    required_keys = {
        "features",
        "labels",
        "label_names",
        "video_paths"
    }


    if not required_keys.issubset(
        set(data.files)
    ):
        raise ValueError(
            f"Missing required keys in {chunk_file}. "
            f"Found: {data.files}"
        )


    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    video_paths = data["video_paths"]


    if features.shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):
        raise ValueError(
            f"Bad feature shape in {chunk_file}: "
            f"{features.shape}"
        )


    sample_count = len(features)


    if not (
        len(labels)
        == sample_count
        == len(label_names)
        == len(video_paths)
    ):
        raise ValueError(
            f"Metadata length mismatch in {chunk_file}"
        )


    feature_parts.append(
        features.astype(
            np.float32,
            copy=False
        )
    )

    label_parts.append(
        labels.astype(
            np.int32,
            copy=False
        )
    )

    label_name_parts.append(
        label_names.astype(str)
    )

    video_path_parts.append(
        video_paths.astype(str)
    )


    if (
        index == 1
        or index % 10 == 0
        or index == len(chunk_files)
    ):
        print(
            f"Loaded {index}/{len(chunk_files)} chunks"
        )


X = np.concatenate(
    feature_parts,
    axis=0
)

y = np.concatenate(
    label_parts,
    axis=0
)

all_label_names = np.concatenate(
    label_name_parts,
    axis=0
)

all_video_paths = np.concatenate(
    video_path_parts,
    axis=0
)


del feature_parts
del label_parts
del label_name_parts
del video_path_parts


print("\nFeatures:", X.shape)
print("Labels:", y.shape)
print("Video paths:", all_video_paths.shape)


# ============================================================
# LOAD OFFICIAL TEST SPLIT
# ============================================================

print("\nLoading official group-aware split...")


split_data = np.load(
    SPLIT_FILE
)


test_indices = split_data[
    "test_indices"
]

split_labels = split_data[
    "labels"
]

split_label_names = split_data[
    "label_names"
]


if len(X) != len(split_labels):
    raise ValueError(
        "Feature count and split metadata count do not match."
    )


if not np.array_equal(
    y,
    split_labels
):
    raise ValueError(
        "Feature labels and split labels do not match."
    )


if not np.array_equal(
    all_label_names.astype(str),
    split_label_names.astype(str)
):
    raise ValueError(
        "Feature label names and split label names do not match."
    )


X_test = X[
    test_indices
]

y_test = y[
    test_indices
]

test_video_paths = all_video_paths[
    test_indices
]


print("\nTest samples:", len(X_test))


# ============================================================
# CLASS MAPPING
# ============================================================

class_mapping = {}


for numeric_label, label_name in zip(
    split_labels,
    split_label_names
):
    class_mapping[
        int(numeric_label)
    ] = str(label_name)


class_ids = sorted(
    class_mapping
)


class_names = [
    class_mapping[class_id]
    for class_id in class_ids
]


print("\nClass mapping:")


for class_id in class_ids:
    print(
        f"{class_id:2} -> "
        f"{class_mapping[class_id]}"
    )


# ============================================================
# LOAD MODEL AND PREDICT
# ============================================================

print("\nLoading EXP-02 model...")


model = keras.models.load_model(
    MODEL_FILE
)


print("Model loaded.")


print("\nPredicting official test set...")


probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


prediction_confidence = np.max(
    probabilities,
    axis=1
)


# ============================================================
# FIND ALL WRONG PREDICTIONS
# ============================================================

wrong_mask = (
    y_pred != y_test
)


wrong_indices = np.where(
    wrong_mask
)[0]


print("\nTotal test samples:", len(y_test))
print("Correct predictions:", int(np.sum(~wrong_mask)))
print("Wrong predictions:", len(wrong_indices))


# ============================================================
# BUILD MISCLASSIFICATION RECORDS
# ============================================================

records = []


for test_position in wrong_indices:

    true_id = int(
        y_test[test_position]
    )

    predicted_id = int(
        y_pred[test_position]
    )

    confidence = float(
        prediction_confidence[test_position]
    )

    original_index = int(
        test_indices[test_position]
    )

    video_path = str(
        test_video_paths[test_position]
    )


    records.append({
        "test_position": int(test_position),
        "original_index": original_index,
        "true_id": true_id,
        "true_class": class_mapping[true_id],
        "predicted_id": predicted_id,
        "predicted_class": class_mapping[predicted_id],
        "confidence": confidence,
        "video_path": video_path
    })


# ============================================================
# COUNT CONFUSION PAIRS
# ============================================================

pair_counts = {}


for record in records:

    pair = (
        record["true_class"],
        record["predicted_class"]
    )

    pair_counts[pair] = (
        pair_counts.get(pair, 0) + 1
    )


sorted_pairs = sorted(
    pair_counts.items(),
    key=lambda item: item[1],
    reverse=True
)


print("\n============================================================")
print("TOP CONFUSION PAIRS")
print("============================================================")


for rank, (
    pair,
    count
) in enumerate(
    sorted_pairs[:TOP_CONFUSION_PAIRS],
    start=1
):

    true_class, predicted_class = pair

    print(
        f"{rank:2}. "
        f"{true_class} -> "
        f"{predicted_class}: "
        f"{count}"
    )


# ============================================================
# CREATE CLEAN AUDIT FOLDER
# ============================================================

if os.path.exists(
    AUDIT_FOLDER
):
    shutil.rmtree(
        AUDIT_FOLDER
    )


os.makedirs(
    AUDIT_FOLDER,
    exist_ok=True
)


# ============================================================
# EXPORT HIGH-CONFIDENCE WRONG VIDEOS
# ============================================================

exported_records = []


for rank, (
    pair,
    pair_count
) in enumerate(
    sorted_pairs[:TOP_CONFUSION_PAIRS],
    start=1
):

    true_class, predicted_class = pair


    pair_records = [
        record
        for record in records
        if (
            record["true_class"] == true_class
            and
            record["predicted_class"] == predicted_class
        )
    ]


    pair_records.sort(
        key=lambda record: record["confidence"],
        reverse=True
    )


    selected_records = pair_records[
        :VIDEOS_PER_PAIR
    ]


    safe_true = (
        true_class
        .replace(" ", "_")
        .replace("/", "_")
    )

    safe_predicted = (
        predicted_class
        .replace(" ", "_")
        .replace("/", "_")
    )


    folder_name = (
        f"{rank:02d}_"
        f"{safe_true}"
        f"_TO_"
        f"{safe_predicted}"
    )


    pair_folder = os.path.join(
        AUDIT_FOLDER,
        folder_name
    )


    os.makedirs(
        pair_folder,
        exist_ok=True
    )


    print(
        f"\nExporting {folder_name}"
    )


    for item_number, record in enumerate(
        selected_records,
        start=1
    ):

        source_path = record[
            "video_path"
        ]


        if not os.path.exists(
            source_path
        ):

            print(
                "MISSING VIDEO:",
                source_path
            )

            continue


        original_filename = os.path.basename(
            source_path
        )


        confidence_percent = (
            record["confidence"] * 100
        )


        destination_filename = (
            f"{item_number:02d}"
            f"__conf_{confidence_percent:06.2f}"
            f"__{original_filename}"
        )


        destination_path = os.path.join(
            pair_folder,
            destination_filename
        )


        shutil.copy2(
            source_path,
            destination_path
        )


        exported_record = dict(
            record
        )

        exported_record[
            "pair_rank"
        ] = rank

        exported_record[
            "pair_total_count"
        ] = pair_count

        exported_record[
            "audit_copy"
        ] = destination_path


        exported_records.append(
            exported_record
        )


        print(
            f"  {item_number:02d}. "
            f"{confidence_percent:6.2f}% | "
            f"{original_filename}"
        )


# ============================================================
# SAVE AUDIT CSV
# ============================================================

csv_fields = [
    "pair_rank",
    "pair_total_count",
    "test_position",
    "original_index",
    "true_id",
    "true_class",
    "predicted_id",
    "predicted_class",
    "confidence",
    "video_path",
    "audit_copy"
]


with open(
    AUDIT_CSV,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=csv_fields
    )

    writer.writeheader()

    writer.writerows(
        exported_records
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("MISCLASSIFICATION AUDIT COMPLETE")
print("============================================================")

print(
    "\nWrong test predictions:",
    len(records)
)

print(
    "Confusion pairs found:",
    len(sorted_pairs)
)

print(
    "Top confusion pairs exported:",
    min(
        TOP_CONFUSION_PAIRS,
        len(sorted_pairs)
    )
)

print(
    "Videos exported:",
    len(exported_records)
)

print("\nAudit folder:")
print(AUDIT_FOLDER)

print("\nAudit CSV:")
print(AUDIT_CSV)

print("\n============================================================")
print("OPEN THE AUDIT FOLDER AND WATCH THE WRONG VIDEOS")
print("============================================================")
