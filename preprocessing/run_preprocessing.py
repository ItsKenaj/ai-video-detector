import os
from preprocessing.extract_frames import extract_frames
from preprocessing.compute_fft import compute_fft
from preprocessing.compute_residuals import compute_residuals

REAL_VIDEOS_DIR = "data/real"
SYNTHETIC_VIDEOS_DIR = "data/synthetic"
FEATURES_DIR = "data/features"


def process_directory(video_dir, label):
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    for video_path in video_paths:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join(FEATURES_DIR, "frames", label, base_name)
        fft_dir = os.path.join(FEATURES_DIR, "fft", label, base_name)
        res_dir = os.path.join(FEATURES_DIR, "residuals", label, base_name)

        print(f"\nProcessing {video_path}")
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
