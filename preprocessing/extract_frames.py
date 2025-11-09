import os
import cv2

def extract_frames(video_path, output_dir, every_n_frames=10):
    """
    Extracts frames from a video file every N frames and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        every_n_frames (int): Interval between saved frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    frame_idx = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path} → {output_dir}")
