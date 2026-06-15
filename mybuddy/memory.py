from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import MyBuddyConfig
from .data import FrameRecord
from .models import CaptionModel, EncodedFrame, FrameEncoder, SummaryModel, Vector, cosine


@dataclass
class MemoryCluster:
    cluster_id: str
    frames: List[EncodedFrame]
    centroid: Vector
    description: str = ""

    @property
    def start(self) -> int:
        return self.frames[0].timestamp if self.frames else 0

    @property
    def end(self) -> int:
        return self.frames[-1].timestamp if self.frames else 0

    def contains_time(self, timestamp: int) -> bool:
        return self.start <= timestamp <= self.end

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "num_frames": len(self.frames),
        }


@dataclass
class LowActiveSummary:

    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, object]:
        return {"start": self.start, "end": self.end, "text": self.text}


class DBSCANCompressor:

    def __init__(self, eps: float = 0.27, min_pts: int = 5):
        self.eps = eps
        self.min_pts = min_pts

    def fit(self, frames: Sequence[EncodedFrame]) -> List[List[int]]:
        n = len(frames)
        if n == 0:
            return []
        labels = [None for _ in range(n)]
        cluster_id = 0
        for idx in range(n):
            if labels[idx] is not None:
                continue
            neighbors = self._region_query(frames, idx)
            if len(neighbors) < self.min_pts:
                labels[idx] = -1
                continue
            cluster_id += 1
            self._expand_cluster(frames, labels, idx, neighbors, cluster_id)
        clusters: Dict[int, List[int]] = {}
        noise: List[int] = []
        for idx, label in enumerate(labels):
            if label == -1 or label is None:
                noise.append(idx)
            else:
                clusters.setdefault(int(label), []).append(idx)
        result = list(clusters.values())
        result.extend([[idx] for idx in noise])
        result.sort(key=lambda ids: frames[ids[0]].timestamp)
        return result

    def _expand_cluster(self, frames: Sequence[EncodedFrame], labels: List[Optional[int]], idx: int, neighbors: List[int], cluster_id: int) -> None:
        labels[idx] = cluster_id
        queue = list(neighbors)
        seen = set(queue)
        while queue:
            current = queue.pop(0)
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] is not None:
                continue
            labels[current] = cluster_id
            current_neighbors = self._region_query(frames, current)
            if len(current_neighbors) >= self.min_pts:
                for nb in current_neighbors:
                    if nb not in seen:
                        queue.append(nb)
                        seen.add(nb)

    def _region_query(self, frames: Sequence[EncodedFrame], idx: int) -> List[int]:
        neighbors: List[int] = []
        emb = frames[idx].embedding
        for j, other in enumerate(frames):
            distance = 1.0 - cosine(emb, other.embedding)
            if distance <= self.eps:
                neighbors.append(j)
        return neighbors


def average_vectors(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0 for _ in range(dim)]
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += value
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


class MultiLevelMemory:

    def __init__(self, cfg: MyBuddyConfig, encoder: Optional[FrameEncoder] = None):
        self.cfg = cfg
        self.encoder = encoder or FrameEncoder(cfg.embedding_dim)
        self.caption_model = CaptionModel()
        self.summary_model = SummaryModel()
        self.compressor = DBSCANCompressor(cfg.dbscan_eps, cfg.dbscan_min_pts)
        self.encoded_frames: List[EncodedFrame] = []
        self.active: List[EncodedFrame] = []
        self.semi_active: List[MemoryCluster] = []
        self.low_active: List[LowActiveSummary] = []
        self.current_time: int = 0
        self._last_summary_end: int = 0

    def reset(self) -> None:
        self.encoded_frames.clear()
        self.active.clear()
        self.semi_active.clear()
        self.low_active.clear()
        self.current_time = 0
        self._last_summary_end = 0

    def update_until(self, frames: Sequence[FrameRecord], timestamp: int) -> None:
        known_times = {f.timestamp for f in self.encoded_frames}
        for frame in frames:
            if frame.timestamp <= timestamp and frame.timestamp not in known_times:
                self.encoded_frames.append(self.encoder.encode_frame(frame))
        self.encoded_frames.sort(key=lambda x: x.timestamp)
        self.current_time = timestamp
        self._refresh_active()
        self._refresh_semi_active()
        self._refresh_low_active()

    def _refresh_active(self) -> None:
        start = self.current_time - self.cfg.active_window
        self.active = [f for f in self.encoded_frames if start <= f.timestamp <= self.current_time]

    def _refresh_semi_active(self) -> None:
        end = self.current_time - self.cfg.active_window
        start = self.current_time - self.cfg.active_window - self.cfg.semi_active_window
        candidates = [f for f in self.encoded_frames if start <= f.timestamp < end]
        clusters: List[MemoryCluster] = []
        for i, ids in enumerate(self.compressor.fit(candidates)):
            frames = [candidates[j] for j in ids]
            centroid = average_vectors([f.embedding for f in frames])
            desc = self.caption_model.describe_cluster(frames)
            clusters.append(MemoryCluster(f"c{i:03d}_{start}_{end}", frames, centroid, desc))
        self.semi_active = clusters

    def _refresh_low_active(self) -> None:
        boundary = self.current_time - self.cfg.active_window - self.cfg.semi_active_window
        if boundary <= self._last_summary_end:
            return
        window = self.cfg.low_active_summary_window
        start = self._last_summary_end
        while start + window <= boundary:
            end = start + window
            frames = [f for f in self.encoded_frames if start <= f.timestamp < end]
            if frames:
                descriptions = self.caption_model.describe_frames(frames)
                text = self.summary_model.summarize(descriptions, start, end)
                self.low_active.append(LowActiveSummary(start, end, text))
            start = end
        self._last_summary_end = start

    def get_active(self) -> List[EncodedFrame]:
        return list(self.active)

    def get_semi_active(self) -> List[MemoryCluster]:
        return list(self.semi_active)

    def get_low_active(self) -> List[LowActiveSummary]:
        return list(self.low_active)

    def describe_state(self) -> Dict[str, object]:
        return {
            "current_time": self.current_time,
            "encoded_frames": len(self.encoded_frames),
            "active_frames": len(self.active),
            "semi_active_clusters": len(self.semi_active),
            "low_active_summaries": len(self.low_active),
            "active_range": self._range(self.active),
            "semi_active_ranges": [(c.start, c.end) for c in self.semi_active],
        }

    def _range(self, frames: Sequence[EncodedFrame]) -> Tuple[Optional[int], Optional[int]]:
        if not frames:
            return None, None
        return frames[0].timestamp, frames[-1].timestamp

    def memory_text(self) -> str:
        lines: List[str] = ["[Active Memory]"]
        lines.extend(f.short_text() for f in self.active[-8:])
        lines.append("[Semi-Active Memory]")
        lines.extend(c.description for c in self.semi_active[-8:])
        lines.append("[Low-Active Memory]")
        lines.extend(s.text for s in self.low_active[-8:])
        return "\n".join(lines)
