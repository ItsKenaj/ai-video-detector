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

# Shuffle once
random.seed(42)
random.shuffle(data)

N = len(data)
train = data[:int(0.70 * N)]
val   = data[int(0.70 * N):int(0.85 * N)]
test  = data[int(0.85 * N):]

def write_file(name, items):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for x in items:
            f.write(x["video_id"] + "\n")

write_file("train_videos.txt", train)
write_file("val_videos.txt", val)
write_file("test_videos.txt", test)

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
