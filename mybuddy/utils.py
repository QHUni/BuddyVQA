import ast
import base64
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(normalize(left), normalize(right)))


def evenly_spaced(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[int(index)] for index in indices]


def image_data_url(path: str | Path, max_side: int) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def parse_mapping(text: str) -> dict:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    raise ValueError(f"Expected a dictionary response, received: {text[:500]}")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

