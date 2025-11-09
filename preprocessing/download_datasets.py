import os
import requests
from datasets import load_dataset


BASE_URL = "https://huggingface.co/datasets/faridlab/deepaction_v1/resolve/main/"
TARGET_DIR = "data/synthetic/deepaction_v1"

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"⚠️ Skipping {url} (HTTP {r.status_code})")

def download_deepaction():
    folders = [
        "BDAnimateDiffLightning",
        "CogVideoX5B",
        "Pexels",
        "RunwayML",
        "StableDiffusion",
        "Veo",
        "VideoPoet",
    ]
    files = ["captions.csv", "README.md"]

    # download root files
    for file in files:
        url = BASE_URL + file
        dest = os.path.join(TARGET_DIR, file)
        download_file(url, dest)

    # just create folder placeholders for now
    for folder in folders:
        os.makedirs(os.path.join(TARGET_DIR, folder), exist_ok=True)
        print(f"📁 Created folder: {folder} (contents must be fetched manually or via HF API)")

def download_kinetics_subset(target_dir="data/real/kinetics_subset"):
    os.makedirs(target_dir, exist_ok=True)
    print("Downloading a small subset of Kinetics (real human videos)...")
    ds = load_dataset("Maysee/kinetics-400-mini", split="train[:10]")
    for i, sample in enumerate(ds):
        video = sample["video"]
        if video is None:
            continue
        with open(os.path.join(target_dir, f"real_{i:03d}.mp4"), "wb") as f:
            f.write(video["bytes"])
    print(" Downloaded small real video subset.")

if __name__ == "__main__":
    download_deepaction()
    download_kinetics_subset()
