import os
import json

FRAME_ROOT = "data/features/frames"

manifest = []

def count_frames(path):
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])

# ---------------------------------------------------------
# REAL VIDEOS
# ---------------------------------------------------------
real_root = os.path.join(FRAME_ROOT, "real")

for subset in ["kinetics_full", "kinetics_mini"]:
    subset_path = os.path.join(real_root, subset)
    if not os.path.isdir(subset_path):
        continue

    for video_id in os.listdir(subset_path):
        video_dir = os.path.join(subset_path, video_id)
        if not os.path.isdir(video_dir):
            continue

        num_frames = count_frames(video_dir)
        if num_frames == 0:
            continue

        manifest.append({
            "video_id": video_dir,
            "label": 0,
            "generator": "real",
            "num_frames": num_frames
        })

# ---------------------------------------------------------
# SYNTHETIC VIDEOS (deepaction_v1)
# ---------------------------------------------------------
syn_root = os.path.join(FRAME_ROOT, "synthetic", "deepaction_v1")

for generator in os.listdir(syn_root):
    gen_path = os.path.join(syn_root, generator)
    if not os.path.isdir(gen_path):
        continue

    for video_id in os.listdir(gen_path):
        video_dir = os.path.join(gen_path, video_id)
        if not os.path.isdir(video_dir):
            continue

        # Handle nested folders like Veo/26/a
        nested_dirs = []
        for sub in os.listdir(video_dir):
            subpath = os.path.join(video_dir, sub)
            if os.path.isdir(subpath):
                nested_dirs.append(subpath)

        # Case A: nested directories (e.g. Veo/26/a)
        if nested_dirs:
            for nd in nested_dirs:
                num_frames = count_frames(nd)
                if num_frames == 0:
                    continue

                manifest.append({
                    "video_id": nd,
                    "label": 1,
                    "generator": generator,
                    "num_frames": num_frames
                })
        else:
            # Case B: frames directly under generator/video_id/
            num_frames = count_frames(video_dir)
            if num_frames > 0:
                manifest.append({
                    "video_id": video_dir,
                    "label": 1,
                    "generator": generator,
                    "num_frames": num_frames
                })

# ---------------------------------------------------------
os.makedirs("splits", exist_ok=True)
with open("splits/video_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Wrote", len(manifest), "video entries to splits/video_manifest.json")
