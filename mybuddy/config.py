from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackendConfig:
    type: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


@dataclass
class EmbedderConfig:
    type: str
    model: str
    device: str = "auto"


@dataclass
class MyBuddyConfig:
    seed: int
    sampling_fps: float
    delta_seconds: float
    gamma_seconds: float
    dbscan_eps: float
    dbscan_min_pts: int
    question_window: int
    top_k_clusters: int
    max_active_frames: int
    max_semi_frames: int
    max_summary_frames: int
    low_summary_window_seconds: float
    low_summary_overlap_seconds: float
    image_max_side: int
    embedder: EmbedderConfig
    small_backend: BackendConfig
    main_backend: BackendConfig
    judge_backend: BackendConfig


def load_config(path: str | Path) -> MyBuddyConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data["embedder"] = EmbedderConfig(**data["embedder"])
    data["small_backend"] = BackendConfig(**data["small_backend"])
    data["main_backend"] = BackendConfig(**data["main_backend"])
    data["judge_backend"] = BackendConfig(**data["judge_backend"])
    return MyBuddyConfig(**data)

