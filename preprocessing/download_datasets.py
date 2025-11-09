import os
from datasets import load_dataset
from huggingface_hub import snapshot_download

REAL_DIR = "data/real/kinetics_mini"
SYNTH_DIR = "data/synthetic/deepaction_v1"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)


def download_small_kinetics(n=5):
    """Download a few small real videos from the Kinetics dataset."""
    print(f"Downloading {n} sample videos from Kinetics (nateraw/kinetics)…")
    ds = load_dataset("nateraw/kinetics", split="train", streaming=True)
    for i, ex in enumerate(ds.take(n)):
        path = os.path.join(REAL_DIR, f"sample_{i}.mp4")
        video_data = ex["video"]
        if isinstance(video_data, (bytes, bytearray)):
            with open(path, "wb") as f:
                f.write(video_data)
        else:
            with open(path, "wb") as f:
                f.write(video_data.read())
        print(f"  • Saved {path}")
    print("✅ Saved real samples to", REAL_DIR)


def download_full_deepaction():
    """Download the full DeepAction v1 dataset (all folders and videos)."""
    print("Downloading full DeepAction v1 dataset (this may take a while)…")
    snapshot_download(
        repo_id="faridlab/deepaction_v1",
        repo_type="dataset",
        local_dir=SYNTH_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("✅ Finished downloading full DeepAction dataset to", SYNTH_DIR)


if __name__ == "__main__":
    print("=== Starting dataset download ===")
    download_small_kinetics(5)
    download_full_deepaction()
    print("\n✅ All datasets downloaded successfully.")
