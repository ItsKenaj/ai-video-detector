import os
from preprocessing.extract_frames import extract_frames
from preprocessing.compute_fft import compute_fft
from preprocessing.compute_residuals import compute_residuals

REAL_VIDEOS_DIR = "data/real"
SYNTHETIC_VIDEOS_DIR = "data/synthetic"
FEATURES_DIR = "data/features"


def process_directory(video_dir, label):
    """
    Recursively process all .mp4 files under video_dir.
    Extract frames, compute FFT, and residual maps.
    """
    for root, _, files in os.walk(video_dir):
        for f in files:
            if not f.endswith(".mp4"):
                continue

            video_path = os.path.join(root, f)

            # create relative path so subfolder structure is preserved under features/
            rel_path = os.path.relpath(root, video_dir)
            base_name = os.path.splitext(f)[0]

            frame_dir = os.path.join(FEATURES_DIR, "frames", label, rel_path, base_name)
            fft_dir = os.path.join(FEATURES_DIR, "fft", label, rel_path, base_name)
            res_dir = os.path.join(FEATURES_DIR, "residuals", label, rel_path, base_name)

            print(f"\nProcessing {video_path}")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(fft_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

            extract_frames(video_path, frame_dir)
            compute_fft(frame_dir, fft_dir)
            compute_residuals(frame_dir, res_dir)


if __name__ == "__main__":
    print("=== Starting Preprocessing Pipeline ===")
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if os.path.exists(REAL_VIDEOS_DIR):
        process_directory(REAL_VIDEOS_DIR, "real")
    if os.path.exists(SYNTHETIC_VIDEOS_DIR):
        process_directory(SYNTHETIC_VIDEOS_DIR, "synthetic")

    print("\nPreprocessing complete. Features stored under data/features/")
