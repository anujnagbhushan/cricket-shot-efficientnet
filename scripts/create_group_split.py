import os
import glob
import re
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

FEATURE_FOLDER = "features"
OUTPUT_FILE = "group_split.npz"
SUMMARY_FILE = "group_split_summary.json"

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE_FROM_REMAINING = 0.1764705882
# 15 / 85 ≈ 0.17647
# Final target is approximately:
# 70% train, 15% validation, 15% test


print("\n============================================================")
print("EXP-02 GROUP-AWARE DATA SPLIT")
print("============================================================")


# ============================================================
# LOAD SAMPLE METADATA
# ============================================================

chunk_files = sorted(
    glob.glob(
        os.path.join(
            FEATURE_FOLDER,
            "chunk_*.npz"
        )
    )
)

all_labels = []
all_label_names = []
all_video_paths = []


for chunk_file in chunk_files:

    data = np.load(chunk_file)

    all_labels.extend(
        data["labels"].tolist()
    )

    all_label_names.extend(
        data["label_names"].tolist()
    )

    all_video_paths.extend(
        data["video_paths"].tolist()
    )


labels = np.asarray(
    all_labels,
    dtype=np.int32
)

label_names = np.asarray(
    all_label_names
)

video_paths = np.asarray(
    all_video_paths
)


# ============================================================
# INFER SOURCE GROUP
# ============================================================

def get_group_id(video_path):

    filename = os.path.basename(video_path)

    filename_without_extension = os.path.splitext(
        filename
    )[0]

    match = re.match(
        r"(.+?)_\d+$",
        filename_without_extension
    )

    if match:

        return match.group(1)

    return filename_without_extension


groups = np.asarray([
    get_group_id(video_path)
    for video_path in video_paths
])


indices = np.arange(
    len(video_paths)
)


print("\nTotal samples:", len(indices))
print("Unique groups:", len(np.unique(groups)))
print("Classes:", len(np.unique(labels)))


# ============================================================
# SPLIT TRAIN+VALIDATION VS TEST
# ============================================================

test_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


train_val_indices, test_indices = next(
    test_splitter.split(
        indices,
        labels,
        groups
    )
)


# ============================================================
# SPLIT TRAIN VS VALIDATION
# ============================================================

remaining_groups = groups[
    train_val_indices
]

remaining_labels = labels[
    train_val_indices
]


validation_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE_FROM_REMAINING,
    random_state=RANDOM_STATE + 1
)


train_relative, validation_relative = next(
    validation_splitter.split(
        train_val_indices,
        remaining_labels,
        remaining_groups
    )
)


train_indices = train_val_indices[
    train_relative
]

validation_indices = train_val_indices[
    validation_relative
]


# ============================================================
# VERIFY GROUP SEPARATION
# ============================================================

train_groups = set(
    groups[train_indices].tolist()
)

validation_groups = set(
    groups[validation_indices].tolist()
)

test_groups = set(
    groups[test_indices].tolist()
)


train_validation_overlap = (
    train_groups & validation_groups
)

train_test_overlap = (
    train_groups & test_groups
)

validation_test_overlap = (
    validation_groups & test_groups
)


# ============================================================
# SUMMARY HELPERS
# ============================================================

def split_summary(split_indices):

    split_labels = label_names[
        split_indices
    ]

    counts = Counter(
        split_labels.tolist()
    )

    return {
        class_name: int(counts[class_name])
        for class_name in sorted(
            set(label_names.tolist())
        )
    }


train_distribution = split_summary(
    train_indices
)

validation_distribution = split_summary(
    validation_indices
)

test_distribution = split_summary(
    test_indices
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n============================================================")
print("SPLIT SUMMARY")
print("============================================================")

print(
    "\nTRAIN:",
    len(train_indices),
    "samples |",
    len(train_groups),
    "groups |",
    f"{len(train_indices) / len(indices) * 100:.2f}%"
)

print(
    "VALIDATION:",
    len(validation_indices),
    "samples |",
    len(validation_groups),
    "groups |",
    f"{len(validation_indices) / len(indices) * 100:.2f}%"
)

print(
    "TEST:",
    len(test_indices),
    "samples |",
    len(test_groups),
    "groups |",
    f"{len(test_indices) / len(indices) * 100:.2f}%"
)


print("\nGROUP OVERLAP CHECK")

print(
    "Train vs Validation:",
    len(train_validation_overlap)
)

print(
    "Train vs Test:",
    len(train_test_overlap)
)

print(
    "Validation vs Test:",
    len(validation_test_overlap)
)


class_names = sorted(
    set(label_names.tolist())
)


print("\nCLASS DISTRIBUTION BY SPLIT")

print(
    f"{'Class':20}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
)

print("-" * 50)


for class_name in class_names:

    print(
        f"{class_name:20}"
        f"{train_distribution[class_name]:10}"
        f"{validation_distribution[class_name]:10}"
        f"{test_distribution[class_name]:10}"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

healthy = True


if train_validation_overlap:

    healthy = False


if train_test_overlap:

    healthy = False


if validation_test_overlap:

    healthy = False


for class_name in class_names:

    if train_distribution[class_name] == 0:

        print(
            "\nWARNING:",
            class_name,
            "missing from TRAIN"
        )

        healthy = False


    if validation_distribution[class_name] == 0:

        print(
            "\nWARNING:",
            class_name,
            "missing from VALIDATION"
        )

        healthy = False


    if test_distribution[class_name] == 0:

        print(
            "\nWARNING:",
            class_name,
            "missing from TEST"
        )

        healthy = False


# ============================================================
# SAVE SPLIT
# ============================================================

np.savez(
    OUTPUT_FILE,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    groups=groups,
    labels=labels,
    label_names=label_names,
    video_paths=video_paths
)


summary = {
    "random_state": RANDOM_STATE,
    "total_samples": int(len(indices)),
    "total_groups": int(len(np.unique(groups))),
    "train_samples": int(len(train_indices)),
    "validation_samples": int(len(validation_indices)),
    "test_samples": int(len(test_indices)),
    "train_groups": int(len(train_groups)),
    "validation_groups": int(len(validation_groups)),
    "test_groups": int(len(test_groups)),
    "train_validation_group_overlap": int(
        len(train_validation_overlap)
    ),
    "train_test_group_overlap": int(
        len(train_test_overlap)
    ),
    "validation_test_group_overlap": int(
        len(validation_test_overlap)
    ),
    "train_distribution": train_distribution,
    "validation_distribution": validation_distribution,
    "test_distribution": test_distribution
}


with open(
    SUMMARY_FILE,
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print("\nSaved split:", OUTPUT_FILE)
print("Saved summary:", SUMMARY_FILE)


print("\n============================================================")

if healthy:

    print(
        "STATUS: GROUP-AWARE SPLIT CREATED SUCCESSFULLY"
    )

else:

    print(
        "STATUS: SPLIT NEEDS INVESTIGATION"
    )

print("============================================================")
