import os, time
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

REAL_DIR = "data/real/kinetics_mini"
SYNTH_DIR = "data/synthetic/deepaction_v1"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)


def download_small_kinetics(n=5):
    print(f"Downloading {n} sample videos from Kinetics (nateraw/kinetics)…")
    ds = load_dataset("nateraw/kinetics", split="train", streaming=True)
    for i, ex in enumerate(ds.take(n)):
        path = os.path.join(REAL_DIR, f"sample_{i}.mp4")
        video_data = ex["video"]
        with open(path, "wb") as f:
            f.write(video_data if isinstance(video_data, (bytes, bytearray)) else video_data.read())
        print(f"  • Saved {path}")
    print("✅ Saved real samples to", REAL_DIR)


def download_some_deepaction(max_per_source=3):
    """Grab a few .mp4s from each DeepAction subfolder to avoid rate limits."""
    print(f"Downloading up to {max_per_source} videos per source from DeepAction v1 …")
    sources = [
        "BDAnimateDiffLightning",
        "CogVideoX5B",
        "Pexels",
        "RunwayML",
        "StableDiffusion",
        "Veo",
        "VideoPoet",
    ]
    repo = "faridlab/deepaction_v1"

    # List all files once to avoid thousands of HTTP calls
    all_files = list_repo_files(repo_id=repo, repo_type="dataset")
    for src in sources:
        src_files = [f for f in all_files if f.startswith(src) and f.endswith(".mp4")]
        if not src_files:
            print(f"  • No mp4s found in {src}, skipping.")
            continue
        for fpath in src_files[:max_per_source]:
            try:
                file_path = hf_hub_download(repo_id=repo, filename=fpath, repo_type="dataset")
                target = os.path.join(SYNTH_DIR, fpath)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.system(f"cp {file_path} {target}")
                print(f"  • Fetched {fpath}")
                time.sleep(0.5)  # throttle slightly to avoid 429
            except Exception as e:
                print(f"  • Skipped {fpath}: {e}")
    print("✅ Finished partial DeepAction download →", SYNTH_DIR)


if __name__ == "__main__":
    print("=== Starting balanced dataset download ===")
    download_small_kinetics(5)
    download_some_deepaction(max_per_source=1)
    print("\n✅ Finished demo download — real + synthetic samples ready.")
