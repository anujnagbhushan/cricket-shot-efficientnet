import os
import glob
import json
import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "EXP-02_EFFICIENTNETB0_GRU"

FEATURE_FOLDER = "features"
SPLIT_FILE = "group_split.npz"

MODEL_FILE = "exp02_efficientnetb0_gru.keras"
HISTORY_FILE = "exp02_training_history.json"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1280

BATCH_SIZE = 32
EPOCHS = 50

RANDOM_STATE = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


print("\n============================================================")
print("EXP-02 EFFICIENTNETB0 + GRU TRAINING")
print("============================================================")

print("\nExperiment:", EXPERIMENT_NAME)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD ALL FEATURE CHUNKS
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


print("\nFeature array:", X.shape)
print("Label array:", y.shape)
print(
    "Feature memory:",
    f"{X.nbytes / (1024 ** 3):.2f} GB"
)


# ============================================================
# LOAD GROUP-AWARE SPLIT
# ============================================================

print("\nLoading group-aware split...")


split_data = np.load(
    SPLIT_FILE
)


train_indices = split_data[
    "train_indices"
]

validation_indices = split_data[
    "validation_indices"
]

test_indices = split_data[
    "test_indices"
]


split_labels = split_data[
    "labels"
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


print(
    "Train samples:",
    len(train_indices)
)

print(
    "Validation samples:",
    len(validation_indices)
)

print(
    "Test samples:",
    len(test_indices)
)


# ============================================================
# CREATE TRAIN / VALIDATION ARRAYS
# ============================================================

print("\nPreparing training and validation data...")


X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


X_validation = X[
    validation_indices
]

y_validation = y[
    validation_indices
]


print(
    "X_train:",
    X_train.shape
)

print(
    "X_validation:",
    X_validation.shape
)


# ============================================================
# CLASS INFORMATION
# ============================================================

label_names = split_data[
    "label_names"
]


class_mapping = {}


for numeric_label, label_name in zip(
    split_labels,
    label_names
):

    class_mapping[
        int(numeric_label)
    ] = str(label_name)


class_names = [
    class_mapping[index]
    for index in sorted(
        class_mapping
    )
]


NUM_CLASSES = len(
    class_names
)


print(
    "\nNumber of classes:",
    NUM_CLASSES
)


print("\nClass mapping:")


for index, class_name in enumerate(
    class_names
):

    print(
        f"{index:2} -> {class_name}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")


unique_classes = np.unique(
    y_train
)


weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)


class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        unique_classes,
        weights
    )
}


for class_id in sorted(
    class_weights
):

    print(
        f"{class_names[class_id]:20} "
        f"{class_weights[class_id]:.4f}"
    )


# ============================================================
# BUILD GRU MODEL
# ============================================================

print("\nBuilding GRU model...")


model = keras.Sequential([
    layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        )
    ),

    layers.GRU(
        256,
        return_sequences=True
    ),

    layers.Dropout(
        0.4
    ),

    layers.GRU(
        128
    ),

    layers.Dropout(
        0.4
    ),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n============================================================")
print("TRAINING START")
print("============================================================")


history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_serializable = {
    key: [
        float(value)
        for value in values
    ]

    for key, values in history.history.items()
}


with open(
    HISTORY_FILE,
    "w"
) as file:

    json.dump(
        history_serializable,
        file,
        indent=4
    )


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

print("\nLoading best saved model...")


best_model = keras.models.load_model(
    MODEL_FILE
)


validation_loss, validation_accuracy = (
    best_model.evaluate(
        X_validation,
        y_validation,
        batch_size=BATCH_SIZE,
        verbose=1
    )
)


print("\n============================================================")
print("TRAINING COMPLETE")
print("============================================================")

print(
    "Best Validation Loss:",
    f"{validation_loss:.6f}"
)

print(
    "Best Validation Accuracy:",
    f"{validation_accuracy:.4f}",
    f"({validation_accuracy * 100:.2f}%)"
)

print("\nModel saved:")
print(MODEL_FILE)

print("\nTraining history saved:")
print(HISTORY_FILE)

print("\nIMPORTANT:")
print(
    "TEST SET HAS NOT BEEN USED FOR TRAINING OR EVALUATION."
)

print("============================================================")
