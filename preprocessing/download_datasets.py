import os
import time
import shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

REAL_DIR = "data/real/kinetics_full"
SYNTH_DIR = "data/synthetic/deepaction_v1"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)


def check_disk(min_free_gb=10):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024 ** 3)
    print(f"Free space available: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        raise RuntimeError(f"Insufficient space (<{min_free_gb} GB). Aborting download.")


def download_real_kinetics(n=50):
    print(f"Downloading {n} real videos from nateraw/kinetics …")
    ds = load_dataset("nateraw/kinetics", split="train", streaming=True)
    for i, ex in enumerate(ds.take(n)):
        path = os.path.join(REAL_DIR, f"sample_{i:03d}.mp4")
        data = ex["video"]
        with open(path, "wb") as f:
            f.write(data if isinstance(data, (bytes, bytearray)) else data.read())
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{n} videos …")
    print(f"Finished downloading {n} real videos → {REAL_DIR}")


def download_some_deepaction(max_per_source=2):
    print(f"Downloading up to {max_per_source} synthetic videos per source from DeepAction v1 …")
    repo = "faridlab/deepaction_v1"
    sources = [
        "BDAnimateDiffLightning",
        "CogVideoX5B",
        "Pexels",
        "RunwayML",
        "StableDiffusion",
        "Veo",
        "VideoPoet",
    ]
    try:
        all_files = list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as e:
        print(f"Could not list repo files: {e}")
        return

    for src in sources:
        mp4s = [f for f in all_files if f.startswith(src) and f.endswith(".mp4")]
        if not mp4s:
            print(f"  No mp4s found in {src}, skipping.")
            continue
        for fpath in mp4s[:max_per_source]:
            try:
                dest = os.path.join(SYNTH_DIR, fpath)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                local = hf_hub_download(repo_id=repo, filename=fpath, repo_type="dataset")
                os.system(f"cp {local} {dest}")
                print(f"  Fetched {fpath}")
                time.sleep(0.3)
            except Exception as e:
                print(f"  Skipped {fpath}: {e}")
    print(f"Finished downloading DeepAction samples → {SYNTH_DIR}")


if __name__ == "__main__":
    print("=== Starting fresh dataset download ===")
    check_disk(min_free_gb=10)
    download_real_kinetics(n=50)
    download_some_deepaction(max_per_source=2)
    print("\nDataset download complete — ready for preprocessing.")
