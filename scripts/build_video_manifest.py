import os
import json

FRAME_ROOT = "data/features/frames"

manifest = []

# -----------------------------------------------------------
# Helper: does this directory contain at least 1 JPG frame?
# -----------------------------------------------------------
def contains_frames(path):
    try:
        for f in os.listdir(path):
            if f.endswith(".jpg"):
                return True
    except:
        return False
    return False


# -----------------------------------------------------------
# Helper: recursively find all "video" directories (folders containing .jpg)
# -----------------------------------------------------------
def find_video_dirs(root):
    video_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # If directory contains jpg frames → it's a video folder
        if any(f.endswith(".jpg") for f in filenames):
            video_dirs.append(dirpath)
    return video_dirs


# -----------------------------------------------------------
# REAL VIDEOS
# -----------------------------------------------------------
real_root = os.path.join(FRAME_ROOT, "real")

real_subsets = [
    os.path.join(real_root, "kinetics_full"),
    os.path.join(real_root, "kinetics_mini"),
]

for subset in real_subsets:
    if not os.path.isdir(subset):
        continue
    video_dirs = find_video_dirs(subset)
    for vd in video_dirs:
        manifest.append({
            "video_id": vd,
            "label": 0,
            "generator": "real",
            "num_frames": len([f for f in os.listdir(vd) if f.endswith(".jpg")]),
        })


# -----------------------------------------------------------
# SYNTHETIC VIDEOS
# -----------------------------------------------------------
synthetic_root = os.path.join(FRAME_ROOT, "synthetic", "deepaction_v1")

# Explicit list of known generators (from your directory listing)
generators = [
    "BDAnimateDiffLightning",
    "CogVideoX5B",
    "Pexels",
    "RunwayML",
    "StableDiffusion",
    "Veo",
    "VideoPoet"
]

for gen in generators:
    gen_path = os.path.join(synthetic_root, gen)
    if not os.path.isdir(gen_path):
        continue

    video_dirs = find_video_dirs(gen_path)
    for vd in video_dirs:
        manifest.append({
            "video_id": vd,
            "label": 1,
            "generator": gen,
            "num_frames": len([f for f in os.listdir(vd) if f.endswith(".jpg")]),
        })


# -----------------------------------------------------------
# Remove duplicates once
# -----------------------------------------------------------
unique = {}
for entry in manifest:
    unique[entry["video_id"]] = entry

final_manifest = list(unique.values())

print("Real videos:", len([m for m in final_manifest if m["label"] == 0]))
print("Synthetic videos:", len([m for m in final_manifest if m["label"] == 1]))
print("TOTAL videos:", len(final_manifest))

os.makedirs("splits", exist_ok=True)
with open("splits/video_manifest.json", "w") as f:
    json.dump(final_manifest, f, indent=2)

print("Wrote splits/video_manifest.json")
