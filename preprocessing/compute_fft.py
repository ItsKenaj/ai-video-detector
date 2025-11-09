import os
import cv2
import numpy as np

def compute_fft(input_dir, output_dir):
    """
    Computes FFT magnitude maps for all images in a directory.

    Args:
        input_dir (str): Folder with extracted frames.
        output_dir (str): Folder to save FFT numpy arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

    if not frame_files:
        print(f" No frames found in {input_dir}")
        return

    for frame_name in frame_files:
        img_path = os.path.join(input_dir, frame_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)

        save_path = os.path.join(output_dir, frame_name.replace(".jpg", "_fft.npy"))
        np.save(save_path, magnitude_spectrum)

    print(f"FFT features saved to {output_dir}")
