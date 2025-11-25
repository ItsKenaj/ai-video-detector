import os
import json

FRAME_ROOT = "data/features/frames"

manifest = []


def add_entries(root, label, generator=None):
    """
    root: folder containing video_id subfolders
    label: 0=real, 1=synthetic
    generator: for synthetic sources
    """
    for video_id in os.listdir(root):
        video_path = os.path.join(root, video_id)
        if not os.path.isdir(video_path):
            continue

        # count frames in this video folder
        frames = [f for f in os.listdir(video_path) if f.endswith(".jpg")]
        num_frames = len(frames)

        if num_frames == 0:
            continue

        entry = {
            "video_id": video_path,  # full path, NOT name only
            "label": label,
            "generator": generator if label == 1 else "real",
            "num_frames": num_frames
        }
        manifest.append(entry)


# -------------------------------------------------------------------
# REAL
# -------------------------------------------------------------------
real_root = os.path.join(FRAME_ROOT, "real")

# kinetics_full
kf_root = os.path.join(real_root, "kinetics_full")
for vid in os.listdir(kf_root):
    add_entries(os.path.join(kf_root), label=0)

# kinetics_mini
km_root = os.path.join(real_root, "kinetics_mini")
for vid in os.listdir(km_root):
    add_entries(os.path.join(km_root), label=0)

# -------------------------------------------------------------------
# SYNTHETIC
# -------------------------------------------------------------------
syn_root = os.path.join(FRAME_ROOT, "synthetic/deepaction_v1")

# generators: Veo, RunwayML, Pexels, StableDiffusion, BDAnimateDiffLightning, ...
for generator in os.listdir(syn_root):
    gen_path = os.path.join(syn_root, generator)
    if not os.path.isdir(gen_path):
        continue

    # inside each generator folder, video_id folders
    for video_id in os.listdir(gen_path):
        video_path = os.path.join(gen_path, video_id)
        if not os.path.isdir(video_path):
            continue

        # some generators have nested structure (example: Veo/26/a/)
        for sub in os.listdir(video_path):
            nested = os.path.join(video_path, sub)
            if os.path.isdir(nested):
                add_entries(nested, label=1, generator=generator)
            else:
                # generator folder directly contains frames
                add_entries(video_path, label=1, generator=generator)

# -------------------------------------------------------------------

os.makedirs("splits", exist_ok=True)
with open("splits/video_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Wrote {len(manifest)} video entries to splits/video_manifest.json")
