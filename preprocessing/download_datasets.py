import os
import requests
from tqdm import tqdm

def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

def download_deepaction():
    base_url = "https://huggingface.co/datasets/faridlab/deepaction_v1/resolve/main/"
    files = [
        "metadata.csv",  # replace with the actual filenames once you inspect the dataset card
    ]
    target_dir = "data/synthetic/deepaction_v1"
    os.makedirs(target_dir, exist_ok=True)
    for file in files:
        url = base_url + file
        dest = os.path.join(target_dir, file)
        download_file(url, dest)
    print(f"✅ DeepAction_v1 dataset downloaded to {target_dir}")

if __name__ == "__main__":
    download_deepaction()
