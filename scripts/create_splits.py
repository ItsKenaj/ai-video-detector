import json
import random
import os

MANIFEST = "splits/video_manifest.json"
OUT_DIR = "splits"

os.makedirs(OUT_DIR, exist_ok=True)

with open(MANIFEST, "r") as f:
    data = json.load(f)

# Remove duplicates by video_id
unique = {}
for item in data:
    unique[item["video_id"]] = item

data = list(unique.values())
print("Unique videos:", len(data))

# -----------------------------------------------------------
# Stratified split: ensure balanced real/synthetic in each set
# -----------------------------------------------------------
real_videos = [v for v in data if v["label"] == 0]
synthetic_videos = [v for v in data if v["label"] == 1]

random.seed(42)
random.shuffle(real_videos)
random.shuffle(synthetic_videos)

def stratified_split(items, train_ratio=0.70, val_ratio=0.15):
    """Split items into train/val/test with given ratios."""
    n = len(items)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    return items[:train_end], items[train_end:val_end], items[val_end:]

real_train, real_val, real_test = stratified_split(real_videos)
syn_train, syn_val, syn_test = stratified_split(synthetic_videos)

# Combine and shuffle within each split
train = real_train + syn_train
val = real_val + syn_val
test = real_test + syn_test

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

def write_file(name, items):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for x in items:
            f.write(x["video_id"] + "\n")

write_file("train_videos.txt", train)
write_file("val_videos.txt", val)
write_file("test_videos.txt", test)

# Print split statistics
def count_labels(items):
    real = sum(1 for x in items if x["label"] == 0)
    syn = sum(1 for x in items if x["label"] == 1)
    return real, syn

print(f"\nTrain: {len(train)} videos (real: {count_labels(train)[0]}, synthetic: {count_labels(train)[1]})")
print(f"Val:   {len(val)} videos (real: {count_labels(val)[0]}, synthetic: {count_labels(val)[1]})")
print(f"Test:  {len(test)} videos (real: {count_labels(test)[0]}, synthetic: {count_labels(test)[1]})")
