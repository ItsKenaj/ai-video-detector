import json
import random
import os

MANIFEST_PATH = "splits/video_manifest.json"
OUTPUT_DIR = "splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load manifest
with open(MANIFEST_PATH, "r") as f:
    data = json.load(f)

# Shuffle deterministically
random.seed(42)
random.shuffle(data)

N = len(data)
train_end = int(0.70 * N)
val_end = int(0.85 * N)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

def write_list(filename, subset):
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for item in subset:
            f.write(item["video_id"] + "\n")

write_list("train_videos.txt", train)
write_list("val_videos.txt", val)
write_list("test_videos.txt", test)

print("Total videos:", N)
print("Train videos:", len(train))
print("Val videos:", len(val))
print("Test videos:", len(test))
print("Wrote split files to:", OUTPUT_DIR)
