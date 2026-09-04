from pathlib import Path
from typing import Iterator

import cv2


class VideoFrameStream:
    def __init__(self, video_path: str | Path, cache_dir: str | Path, sampling_fps: float):
        self.video_path = Path(video_path)
        self.cache_dir = Path(cache_dir)
        self.sampling_fps = sampling_fps
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[tuple[float, Path]]:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            source_fps = 30.0
        duration_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = duration_frames / source_fps if duration_frames > 0 else float("inf")
        step = 1.0 / self.sampling_fps
        sample_index = 0
        timestamp = 0.0
        try:
            while timestamp <= duration + 1e-6:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    break
                target = self.cache_dir / f"frame_{sample_index:07d}_{timestamp:010.3f}.jpg"
                if not target.exists():
                    if not cv2.imwrite(str(target), frame):
                        raise RuntimeError(f"Unable to write frame: {target}")
                yield timestamp, target
                sample_index += 1
                timestamp = sample_index * step
        finally:
            capture.release()

