import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (320, 240))
    for index in range(950):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = ((40 + index) % 256, 80, 140)
        cv2.circle(frame, (40 + index % 240, 120), 25, (20, 220, 220), -1)
        cv2.putText(frame, f"t={index / 5:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    records = [
        {"video_id": "demo", "video_path": "demo.mp4", "qid": "q1", "timestamp": 8.0, "question": "What is visible?", "answers": ["unknown"], "category": "Recognition", "chain": False, "deictic": False},
        {"video_id": "demo", "video_path": "demo.mp4", "qid": "q2", "timestamp": 80.0, "question": "Is it still there?", "answers": ["unknown"], "category": "Recall", "chain": True, "deictic": True},
        {"video_id": "demo", "video_path": "demo.mp4", "qid": "q3", "timestamp": 160.0, "question": "What should I do next?", "answers": ["unknown"], "category": "Advisory", "chain": True, "deictic": False},
        {"video_id": "demo", "video_path": "demo.mp4", "qid": "q4", "timestamp": 185.0, "question": "Did I do it correctly?", "answers": ["unknown"], "category": "Scrutinization", "chain": True, "deictic": True}
    ]
    with (output / "demo.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
