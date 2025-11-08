"""
Download and prepare datasets for AI Video Detector.
Supports Hugging Face and ModelScope sources.
"""

from datasets import load_dataset
from modelscope.msdatasets import MsDataset
from pathlib import Path

def download_huggingface(dataset_name, save_dir):
    print(f"Downloading {dataset_name}...")
    ds = load_dataset(dataset_name)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Dataset {dataset_name} available at {save_path} (via cache).")
    return ds

def download_modelscope(dataset_name, save_dir):
    print(f"Downloading {dataset_name}...")
    ds = MsDataset.load(dataset_name)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Dataset {dataset_name} available at {save_path} (via cache).")
    return ds

if __name__ == "__main__":
    download_huggingface("faridlab/deepaction_v1", "data/synthetic/deepaction_v1")
    download_huggingface("kalpitbcontrails/seetrails_aigvdet_v2.0.0", "data/synthetic/seetrails")
    download_modelscope("cccnju/GenVideo-100K", "data/synthetic/genvideo")
