import os
import random
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REAL_DIR = "data/real/kinetics_mini"
SYNTH_DIR = "data/synthetic/deepaction_v1"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)

def download_small_kinetics(n=5):
    print(f"Downloading {n} sample videos from Kinetics-Mini …")
    ds = load_dataset("faridlab/kinetics_mini", split="train", streaming=True)
    for i, ex in enumerate(ds.take(n)):
        path = os.path.join(REAL_DIR, f"sample_{i}.mp4")
        with open(path, "wb") as f:
            f.write(ex["video"].read())
    print("✅ Saved real samples to", REAL_DIR)

def download_small_deepaction(n=5):
    print(f"Downloading {n} sample videos from DeepAction v1 …")
    sources = [
        "BDAnimateDiffLightning",
        "RunwayML",
        "StableDiffusion",
        "VideoPoet",
    ]
    for src in sources[:n]:
        try:
            file_path = hf_hub_download(
                repo_id="faridlab/deepaction_v1",
                filename=f"{src}/sample_0.mp4",
                repo_type="dataset",
            )
            target = os.path.join(SYNTH_DIR, f"{src}_sample.mp4")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.system(f"cp {file_path} {target}")
            print(f"  • Fetched {src}_sample.mp4")
        except Exception as e:
            print(f"  • Skipped {src}: {e}")

if __name__ == "__main__":
    print("=== Starting lightweight dataset download ===")
    download_small_kinetics(5)
    download_small_deepaction(5)
    print("\n✅ Finished demo download — 10 videos total.")
