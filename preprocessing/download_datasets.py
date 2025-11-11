# preprocessing/download_datasets.py
import os
import time
import shutil
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

REAL_DIR = Path("data/real/kinetics_full")
SYNTH_DIR = Path("data/synthetic/deepaction_v1")
DEEPACTION_SOURCES = [
    "BDAnimateDiffLightning",
    "CogVideoX5B",
    "Pexels",
    "RunwayML",
    "StableDiffusion",
    "Veo",
    "VideoPoet",
]

def check_disk(min_free_gb=10.0):
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"[disk] Free space: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        raise RuntimeError(f"Insufficient space (< {min_free_gb} GB). Aborting.")

def download_real_kinetics(n=100, overwrite=False, throttle=0.0):
    """Download n real videos from the full Kinetics dataset."""
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[real] Downloading {n} videos from nateraw/kinetics (streaming)…")

    # use the *full* Kinetics dataset, not kinetics-mini
    ds = load_dataset("nateraw/kinetics", split="train", streaming=True)

    count = 0
    for i, ex in enumerate(ds):
        if count >= n:
            break
        out_path = REAL_DIR / f"sample_{i:04d}.mp4"
        if out_path.exists() and not overwrite:
            continue
        try:
            data = ex["video"]
            with open(out_path, "wb") as f:
                f.write(data if isinstance(data, (bytes, bytearray)) else data.read())
            count += 1
            if count % 10 == 0 or count == n:
                print(f"  [real] saved {count}/{n} → {out_path.name}")
            if throttle > 0:
                time.sleep(throttle)
        except Exception as e:
            print(f"  [real] skip index {i}: {e}")
            continue
    print(f"[real] Finished: {count} videos in {REAL_DIR}")

def download_deepaction(per_source=2, overwrite=False, throttle=0.2, shuffle=True, seed=1337):
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    repo = "faridlab/deepaction_v1"
    print(f"[syn] Listing DeepAction repo files …")
    try:
        all_files = list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as e:
        print(f"[syn] Could not list repo files: {e}")
        return
    rng = random.Random(seed)

    total_fetched = 0
    for src in DEEPACTION_SOURCES:
        mp4s = [f for f in all_files if f.startswith(src + "/") and f.endswith(".mp4")]
        if not mp4s:
            print(f"  [syn] {src}: no mp4s found, skip.")
            continue
        if shuffle:
            rng.shuffle(mp4s)
        chosen = mp4s[:per_source]
        fetched_src = 0
        for relpath in chosen:
            dest = SYNTH_DIR / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not overwrite:
                # already have it, count as fetched to keep balance consistent
                fetched_src += 1
                total_fetched += 1
                continue
            try:
                local = hf_hub_download(repo_id=repo, filename=relpath, repo_type="dataset")
                shutil.copy2(local, dest)
                fetched_src += 1
                total_fetched += 1
                print(f"  [syn] fetched {src}: {Path(relpath).name}")
                if throttle > 0:
                    time.sleep(throttle)
            except Exception as e:
                print(f"  [syn] skip {relpath}: {e}")
        print(f"  [syn] {src}: {fetched_src}/{per_source} saved.")
    print(f"[syn] Finished: {total_fetched} total synthetic videos in {SYNTH_DIR}")

def parse_args():
    p = argparse.ArgumentParser(description="Download real + synthetic videos (balanced).")
    p.add_argument("--n-real", type=int, default=100, help="Number of real Kinetics videos")
    p.add_argument("--per-source", type=int, default=3, help="Synthetic videos per DeepAction source")
    p.add_argument("--min-free-gb", type=float, default=10.0, help="Required free space to proceed")
    p.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    p.add_argument("--throttle", type=float, default=0.2, help="Sleep seconds between downloads")
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle DeepAction file list")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=== Fresh dataset download ===")
    check_disk(args.min_free_gb)
    download_real_kinetics(n=args.n_real, overwrite=args.overwrite, throttle=args.throttle)
    download_deepaction(
        per_source=args.per_source,
        overwrite=args.overwrite,
        throttle=args.throttle,
        shuffle=(not args.no_shuffle),
    )
    print("Dataset download complete — ready for preprocessing.")
