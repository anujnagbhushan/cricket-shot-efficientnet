import os
import glob
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FOLDER = "features_b3"
SPLIT_FILE = "group_split.npz"
MODEL_FILE = "models/exp05_efficientnetb3_gru.keras"

RESULT_FILE = "exp05_test_results.json"
CONFUSION_FILE = "exp05_confusion_matrix.npy"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1536

BATCH_SIZE = 32


print("\n============================================================")
print("EXP-02 FINAL TEST EVALUATION")
print("============================================================")

print("\nTensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD FEATURE CHUNKS
# ============================================================

print("\nLoading EfficientNet feature chunks...")


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


for index, chunk_file in enumerate(
    chunk_files,
    start=1
):

    data = np.load(chunk_file)

    features = data["features"]
    labels = data["labels"]


    if features.shape[1:] != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        raise ValueError(
            f"Bad feature shape in {chunk_file}: "
            f"{features.shape}"
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


del feature_parts
del label_parts


print("\nAll features:", X.shape)
print("All labels:", y.shape)


# ============================================================
# LOAD OFFICIAL GROUP SPLIT
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


label_names = split_data[
    "label_names"
]


groups = split_data[
    "groups"
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


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]


test_groups = np.unique(
    groups[test_indices]
)


print("\nTest samples:", len(X_test))
print("Test groups:", len(test_groups))

print(
    "X_test:",
    X_test.shape
)


# ============================================================
# CLASS MAPPING
# ============================================================

class_mapping = {}


for numeric_label, label_name in zip(
    split_labels,
    label_names
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
# LOAD SELECTED MODEL
# ============================================================

print("\nLoading selected model...")


model = keras.models.load_model(
    MODEL_FILE
)


print("Model loaded successfully.")


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print("\n============================================================")
print("RUNNING FINAL TEST PREDICTIONS")
print("============================================================")


probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


y_pred = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_pred
)


report = classification_report(
    y_test,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)


report_text = classification_report(
    y_test,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    zero_division=0
)


confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=class_ids
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n============================================================")
print("FINAL TEST RESULT")
print("============================================================")


print(
    "\nTest Accuracy:",
    f"{test_accuracy:.4f}",
    f"({test_accuracy * 100:.2f}%)"
)


print("\nCLASSIFICATION REPORT\n")

print(
    report_text
)


print("\nCONFUSION MATRIX")

print(
    confusion
)


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

confusion_pairs = []


for true_index in range(
    len(class_ids)
):

    for predicted_index in range(
        len(class_ids)
    ):

        if true_index == predicted_index:

            continue


        count = int(
            confusion[
                true_index,
                predicted_index
            ]
        )


        if count > 0:

            confusion_pairs.append(
                (
                    count,
                    class_names[true_index],
                    class_names[predicted_index]
                )
            )


confusion_pairs.sort(
    reverse=True
)


print("\n============================================================")
print("TOP 20 CONFUSIONS")
print("============================================================")


for rank, (
    count,
    true_class,
    predicted_class
) in enumerate(
    confusion_pairs[:20],
    start=1
):

    print(
        f"{rank:2}. "
        f"{true_class} -> "
        f"{predicted_class}: "
        f"{count}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

result_data = {
    "model_file": MODEL_FILE,
    "test_samples": int(
        len(X_test)
    ),
    "test_groups": int(
        len(test_groups)
    ),
    "test_accuracy": float(
        test_accuracy
    ),
    "classification_report": report,
    "top_confusions": [
        {
            "count": count,
            "true_class": true_class,
            "predicted_class": predicted_class
        }

        for (
            count,
            true_class,
            predicted_class
        ) in confusion_pairs[:20]
    ]
}


with open(
    RESULT_FILE,
    "w"
) as file:

    json.dump(
        result_data,
        file,
        indent=4
    )


np.save(
    CONFUSION_FILE,
    confusion
)


print("\nSaved test results:")
print(RESULT_FILE)

print("\nSaved confusion matrix:")
print(CONFUSION_FILE)


print("\n============================================================")
print("EXP-02 FINAL TEST EVALUATION COMPLETE")
print("============================================================")
