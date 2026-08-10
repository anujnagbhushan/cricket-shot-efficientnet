import os
import glob
import re
import numpy as np
from collections import Counter, defaultdict

FEATURE_FOLDER = "features"

print("\n============================================================")
print("EXP-02 VIDEO GROUP AUDIT")
print("============================================================")

chunk_files = sorted(
    glob.glob(
        os.path.join(
            FEATURE_FOLDER,
            "chunk_*.npz"
        )
    )
)

all_paths = []
all_labels = []

for chunk_file in chunk_files:

    data = np.load(chunk_file)

    all_paths.extend(
        data["video_paths"].tolist()
    )

    all_labels.extend(
        data["label_names"].tolist()
    )


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


groups = defaultdict(list)

for video_path, label in zip(
    all_paths,
    all_labels
):

    group_id = get_group_id(video_path)

    groups[group_id].append(
        (
            video_path,
            label
        )
    )


group_sizes = Counter(
    len(videos)
    for videos in groups.values()
)


multi_clip_groups = {
    group_id: videos
    for group_id, videos in groups.items()
    if len(videos) > 1
}


mixed_label_groups = {}

for group_id, videos in groups.items():

    labels = set(
        label
        for _, label in videos
    )

    if len(labels) > 1:

        mixed_label_groups[group_id] = videos


print("\nTotal video samples:", len(all_paths))
print("Unique inferred source groups:", len(groups))

print(
    "Groups containing multiple clips:",
    len(multi_clip_groups)
)

print(
    "Groups containing multiple class labels:",
    len(mixed_label_groups)
)


print("\nGROUP SIZE DISTRIBUTION")

for size, count in sorted(
    group_sizes.items()
):

    print(
        f"{size:4} clips/group : "
        f"{count:5} groups"
    )


print("\n============================================================")
print("LARGEST 20 GROUPS")
print("============================================================")

largest_groups = sorted(
    groups.items(),
    key=lambda item: len(item[1]),
    reverse=True
)[:20]


for group_id, videos in largest_groups:

    labels = sorted(
        set(
            label
            for _, label in videos
        )
    )

    print(
        f"\nGroup: {group_id}"
    )

    print(
        "Number of clips:",
        len(videos)
    )

    print(
        "Labels:",
        labels
    )

    for video_path, label in videos[:5]:

        print(
            "   ",
            os.path.basename(video_path),
            "->",
            label
        )

    if len(videos) > 5:

        print(
            f"    ... and "
            f"{len(videos) - 5} more"
        )


print("\n============================================================")
print("AUDIT COMPLETE")
print("============================================================")

if len(multi_clip_groups) > 0:

    print(
        "RESULT: MULTIPLE CLIPS SHARE THE SAME INFERRED SOURCE ID"
    )

    print(
        "RECOMMENDATION: USE GROUP-AWARE TRAIN/TEST SPLITTING"
    )

else:

    print(
        "RESULT: NO REPEATED INFERRED SOURCE GROUPS FOUND"
    )

    print(
        "RECOMMENDATION: STANDARD STRATIFIED SPLIT MAY BE USED"
    )

print("============================================================")
