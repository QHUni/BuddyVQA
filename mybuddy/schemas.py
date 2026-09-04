from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FrameRecord:
    timestamp: float
    path: Path
    embedding: np.ndarray


@dataclass
class MemoryCluster:
    cluster_id: int
    center: np.ndarray
    members: list[FrameRecord] = field(default_factory=list)


@dataclass
class QAPair:
    qid: str
    question: str
    answer: str
    timestamp: float


@dataclass
class Sample:
    video_id: str
    video_path: Path
    qid: str
    timestamp: float
    question: str
    answers: list[str]
    metadata: dict


@dataclass
class Prediction:
    video_id: str
    qid: str
    timestamp: float
    question: str
    answers: list[str]
    metadata: dict
    rephrased_question: str
    prediction: str
    relevant_qa: dict | None
    visual_descriptions: dict
