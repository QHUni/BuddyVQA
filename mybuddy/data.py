from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


CATEGORIES = ["Recall", "Detection", "Advisory", "Evaluation"]
DEICTIC_WORDS = ["it", "this", "that", "here", "there", "these", "those", "they", "them"]
OBJECTS = ["knife", "phone", "mango", "bottle", "tent", "bag", "shoe", "bucket", "cup", "sauce"]
LOCATIONS = ["counter", "shelf", "table", "trunk", "basket", "corner", "floor", "yard", "sink", "rack"]
ACTIONS = ["cutting", "cleaning", "shopping", "cooking", "packing", "walking", "fixing", "organizing"]


@dataclass
class FrameRecord:

    video_id: str
    timestamp: int
    text: str
    objects: List[str] = field(default_factory=list)
    location: str = "unknown"
    gaze: Tuple[float, float] = (0.5, 0.5)
    hand_bbox: Tuple[float, float, float, float] = (0.35, 0.35, 0.65, 0.65)

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "timestamp": self.timestamp,
            "text": self.text,
            "objects": self.objects,
            "location": self.location,
            "gaze": list(self.gaze),
            "hand_bbox": list(self.hand_bbox),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FrameRecord":
        return cls(
            video_id=str(data["video_id"]),
            timestamp=int(data["timestamp"]),
            text=str(data.get("text", "")),
            objects=list(data.get("objects", [])),
            location=str(data.get("location", "unknown")),
            gaze=tuple(data.get("gaze", (0.5, 0.5))),
            hand_bbox=tuple(data.get("hand_bbox", (0.35, 0.35, 0.65, 0.65))),
        )


@dataclass
class QARecord:
    """A timestamped question-answer record."""

    video_id: str
    qid: str
    timestamp: int
    question: str
    answers: List[str]
    category: str
    chained_to: Optional[str] = None
    highlight_start: Optional[int] = None
    highlight_end: Optional[int] = None

    @property
    def is_chained(self) -> bool:
        return self.chained_to is not None

    @property
    def has_deictic(self) -> bool:
        text = self.question.lower().replace("?", " ")
        tokens = {tok.strip(".,!?;:") for tok in text.split()}
        return any(w in tokens for w in DEICTIC_WORDS)

    def primary_answer(self) -> str:
        return self.answers[0] if self.answers else "unknown"

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "qid": self.qid,
            "timestamp": self.timestamp,
            "question": self.question,
            "answers": self.answers,
            "category": self.category,
            "chained_to": self.chained_to,
            "highlight_start": self.highlight_start,
            "highlight_end": self.highlight_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "QARecord":
        return cls(
            video_id=str(data["video_id"]),
            qid=str(data["qid"]),
            timestamp=int(data["timestamp"]),
            question=str(data["question"]),
            answers=list(data.get("answers", [])),
            category=str(data.get("category", "Detection")),
            chained_to=data.get("chained_to"),
            highlight_start=data.get("highlight_start"),
            highlight_end=data.get("highlight_end"),
        )


@dataclass
class VideoExample:

    video_id: str
    frames: List[FrameRecord]
    qas: List[QARecord]

    def sort(self) -> "VideoExample":
        self.frames.sort(key=lambda f: f.timestamp)
        self.qas.sort(key=lambda q: q.timestamp)
        return self

    def frame_at_or_before(self, timestamp: int) -> FrameRecord:
        candidates = [f for f in self.frames if f.timestamp <= timestamp]
        return candidates[-1] if candidates else self.frames[0]

    def frames_until(self, timestamp: int) -> List[FrameRecord]:
        return [f for f in self.frames if f.timestamp <= timestamp]

    def qa_by_id(self) -> Dict[str, QARecord]:
        return {q.qid: q for q in self.qas}

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "frames": [f.to_dict() for f in self.frames],
            "qas": [q.to_dict() for q in self.qas],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "VideoExample":
        frames = [FrameRecord.from_dict(x) for x in data.get("frames", [])]
        qas = [QARecord.from_dict(x) for x in data.get("qas", [])]
        return cls(video_id=str(data["video_id"]), frames=frames, qas=qas).sort()


def tokenize(text: str) -> List[str]:
    return [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split() if t.strip()]


def contains_deictic(text: str) -> bool:
    tokens = set(tokenize(text))
    return any(w in tokens for w in DEICTIC_WORDS)


def load_jsonl(path: str) -> List[VideoExample]:
    videos: List[VideoExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            videos.append(VideoExample.from_dict(json.loads(line)))
    return videos


def save_jsonl(videos: Sequence[VideoExample], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for video in videos:
            f.write(json.dumps(video.to_dict(), ensure_ascii=False) + "\n")


def _make_frame(video_id: str, t: int, rng: random.Random) -> FrameRecord:
    obj1 = rng.choice(OBJECTS)
    obj2 = rng.choice([o for o in OBJECTS if o != obj1])
    loc = rng.choice(LOCATIONS)
    act = rng.choice(ACTIONS)
    text = f"The wearer is {act} near the {loc}; visible objects include {obj1} and {obj2}."
    gaze = (round(rng.uniform(0.2, 0.8), 3), round(rng.uniform(0.2, 0.8), 3))
    w, h = rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.35)
    x1 = max(0.0, gaze[0] - w / 2)
    y1 = max(0.0, gaze[1] - h / 2)
    x2 = min(1.0, gaze[0] + w / 2)
    y2 = min(1.0, gaze[1] + h / 2)
    return FrameRecord(video_id, t, text, [obj1, obj2], loc, gaze, (x1, y1, x2, y2))


def _question_templates(obj: str, loc: str, category: str, deictic: bool, chained: bool) -> Tuple[str, List[str]]:
    target = "it" if deictic else f"the {obj}"
    if category == "Recall":
        q = f"Where did I put {target}?" if not chained else f"Where is {target} now?"
        a = [f"The {obj} is near the {loc}.", f"It appears to be around the {loc}."]
    elif category == "Detection":
        q = f"What is next to {target}?" if deictic else f"What is next to the {obj}?"
        a = [f"Another visible item is next to the {obj} near the {loc}."]
    elif category == "Advisory":
        q = f"How should I use {target}?" if deictic else f"How should I use the {obj}?"
        a = [f"Use the {obj} carefully and keep it stable near the {loc}.", f"Keep the area around the {loc} organized first."]
    else:
        q = f"Is {target} safe here?" if deictic else f"Is using the {obj} safe here?"
        a = [f"It seems mostly safe if the {obj} is handled carefully near the {loc}."]
    return q, a


def generate_synthetic_dataset(num_videos: int = 16, seed: int = 42, qas_per_video: int = 6) -> List[VideoExample]:
    rng = random.Random(seed)
    videos: List[VideoExample] = []
    for vid in range(num_videos):
        video_id = f"synthetic_{vid:04d}"
        duration = rng.randint(90, 180)
        frames = [_make_frame(video_id, t, rng) for t in range(0, duration + 1, 5)]
        qas: List[QARecord] = []
        times = sorted(rng.sample(range(10, duration - 5), k=min(qas_per_video, duration - 20)))
        for i, ts in enumerate(times):
            frame = max([f for f in frames if f.timestamp <= ts], key=lambda f: f.timestamp)
            category = rng.choice(CATEGORIES)
            deictic = rng.random() < 0.82
            chained = i > 0 and rng.random() < 0.72
            chained_to = rng.choice(qas).qid if chained and qas else None
            obj = frame.objects[0]
            q, answers = _question_templates(obj, frame.location, category, deictic, chained)
            if chained_to:
                base = next(x for x in qas if x.qid == chained_to)
                if contains_deictic(q):
                    answers = [f"Based on the previous answer, {answers[0]}"] + answers[1:]
            qas.append(
                QARecord(
                    video_id=video_id,
                    qid=f"{video_id}_q{i:03d}",
                    timestamp=ts,
                    question=q,
                    answers=answers,
                    category=category,
                    chained_to=chained_to,
                    highlight_start=max(0, ts - rng.randint(3, 10)),
                    highlight_end=min(duration, ts + rng.randint(2, 8)),
                )
            )
        videos.append(VideoExample(video_id, frames, qas).sort())
    return videos


def split_dataset(videos: Sequence[VideoExample], val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[VideoExample], List[VideoExample], List[VideoExample]]:
    rng = random.Random(seed)
    items = list(videos)
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(math.ceil(n * test_ratio)))
    n_val = max(1, int(math.ceil(n * val_ratio)))
    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    return train, val, test


def iter_qas(videos: Iterable[VideoExample]) -> Iterator[Tuple[VideoExample, QARecord]]:
    for video in videos:
        for qa in video.qas:
            yield video, qa
