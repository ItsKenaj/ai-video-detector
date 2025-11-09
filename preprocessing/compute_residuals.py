import os
import cv2
import numpy as np

def compute_residuals(input_dir, output_dir):
    """
    Computes high-frequency residual maps by subtracting a Gaussian-blurred
    version of each frame from the original frame.

    Args:
        input_dir (str): Folder with extracted frames.
        output_dir (str): Folder to save residual maps.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    if not frame_files:
        print(f"No frames found in {input_dir}")
        return

    for frame_name in frame_files:
        img_path = os.path.join(input_dir, frame_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        residual = cv2.absdiff(img, blurred)

        save_path = os.path.join(output_dir, frame_name.replace(".jpg", "_residual.npy"))
        np.save(save_path, residual)

    print(f"Residual maps saved to {output_dir}")
