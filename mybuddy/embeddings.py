from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .config import EmbedderConfig
from .utils import normalize


class Embedder(Protocol):
    def encode_image(self, path: str | Path) -> np.ndarray:
        ...

    def encode_text(self, text: str) -> np.ndarray:
        ...


class HistogramEmbedder:
    def encode_image(self, path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        return normalize(histogram)

    def encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(128, dtype=np.float32)
        encoded = text.lower().encode("utf-8")
        for index, value in enumerate(encoded):
            vector[(value + index * 31) % 128] += 1.0
        return normalize(vector)


class CLIPEmbedder:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        resolved = device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = resolved
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def encode_image(self, path: str | Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vector = self.model.get_image_features(**inputs)[0].float().cpu().numpy()
        return normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vector = self.model.get_text_features(**inputs)[0].float().cpu().numpy()
        return normalize(vector)


def build_embedder(config: EmbedderConfig) -> Embedder:
    if config.type == "histogram":
        return HistogramEmbedder()
    if config.type == "clip":
        return CLIPEmbedder(config.model, config.device)
    raise ValueError(f"Unknown embedder type: {config.type}")

