import torch
import cv2
import numpy as np
from torchvision import models
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(weights_path="results/checkpoints/baseline_resnet18.pt"):
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def preprocess_frame(frame):
    import scipy.fft as fft
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    residual = cv2.Laplacian(gray, cv2.CV_32F)
    f = np.fft.fftshift(np.fft.fft2(gray))
    fft_mag = np.log(np.abs(f) + 1e-8)
    fft_mag = cv2.resize(fft_mag, (224, 224))
    residual = cv2.resize(residual, (224, 224))
    frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    x = np.concatenate([frame, fft_mag[..., None], residual[..., None]], axis=2)
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0).float().to(DEVICE)
    return x

def predict_video(path):
    model = load_model()
    cap = cv2.VideoCapture(path)
    scores = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess_frame(frame)
        with torch.no_grad():
            logit = model(x)
            prob_fake = torch.sigmoid(logit).item()
        scores.append(prob_fake)
    cap.release()
    return np.mean(scores)

if __name__ == "__main__":
    test_video = "data/test/sora_scene.mp4"
    score = predict_video(test_video)
    print(f"Fake probability for {test_video}: {score:.3f}")
