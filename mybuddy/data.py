import json
from pathlib import Path
from typing import Any

from .schemas import Sample


def _records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("data", "videos", "samples", "annotations"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("The dataset JSON must contain a list or one of data/videos/samples/annotations")


def _first(record: dict, keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _answers(record: dict) -> list[str]:
    value = _first(record, ("answers", "reference_answers", "gt_answers", "answer", "gt_answer"), [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_samples(path: str | Path) -> list[Sample]:
    source = Path(path).resolve()
    flat: list[dict] = []
    for record in _records(source):
        questions = record.get("questions")
        if isinstance(questions, list):
            inherited = {key: value for key, value in record.items() if key != "questions"}
            flat.extend([{**inherited, **question} for question in questions])
        else:
            flat.append(record)
    samples: list[Sample] = []
    for index, record in enumerate(flat):
        video_value = _first(record, ("video_path", "video", "video_file", "path"))
        if video_value is None:
            raise ValueError(f"Missing video path in record {index}")
        video_path = Path(str(video_value))
        if not video_path.is_absolute():
            video_path = (source.parent / video_path).resolve()
        video_id = str(_first(record, ("video_id", "vid", "clip_id"), video_path.stem))
        qid = str(_first(record, ("qid", "question_id", "id"), f"{video_id}-{index}"))
        timestamp = float(_first(record, ("timestamp", "question_time", "time", "t"), 0.0))
        question = str(_first(record, ("question", "query", "q"), ""))
        if not question:
            raise ValueError(f"Missing question in record {index}")
        excluded = {"video_path", "video", "video_file", "path", "video_id", "vid", "clip_id", "qid", "question_id", "id", "timestamp", "question_time", "time", "t", "question", "query", "q", "answers", "reference_answers", "gt_answers", "answer", "gt_answer"}
        metadata = {key: value for key, value in record.items() if key not in excluded}
        samples.append(Sample(video_id, video_path, qid, timestamp, question, _answers(record), metadata))
    return sorted(samples, key=lambda item: (item.video_id, item.timestamp, item.qid))

