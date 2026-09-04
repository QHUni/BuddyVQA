from collections import deque
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from .config import MyBuddyConfig
from .embeddings import Embedder
from .llm import MLLMBackend
from .prompts import LOW_ACTIVE_SUMMARY
from .schemas import FrameRecord, MemoryCluster
from .utils import cosine_distance, evenly_spaced, normalize


class IncrementalDensityMemory:
    def __init__(self, eps: float, min_pts: int):
        self.eps = eps
        self.min_pts = min_pts
        self.clusters: dict[int, MemoryCluster] = {}
        self.noise: list[FrameRecord] = []
        self.next_cluster_id = 0

    def _refresh(self, cluster: MemoryCluster) -> None:
        if cluster.members:
            cluster.center = normalize(np.mean([member.embedding for member in cluster.members], axis=0))

    def _add_cluster(self, members: list[FrameRecord]) -> None:
        center = normalize(np.mean([member.embedding for member in members], axis=0))
        cluster = MemoryCluster(self.next_cluster_id, center, list(members))
        self.clusters[cluster.cluster_id] = cluster
        self.next_cluster_id += 1

    def _promote_noise(self) -> None:
        if len(self.noise) < self.min_pts:
            return
        features = np.stack([item.embedding for item in self.noise])
        labels = DBSCAN(eps=self.eps, min_samples=self.min_pts, metric="cosine").fit_predict(features)
        retained: list[FrameRecord] = []
        for label in sorted(set(labels)):
            members = [item for item, value in zip(self.noise, labels) if value == label]
            if label < 0:
                retained.extend(members)
            else:
                self._add_cluster(members)
        self.noise = retained

    def add(self, record: FrameRecord) -> None:
        candidates = sorted(self.clusters.values(), key=lambda cluster: cosine_distance(record.embedding, cluster.center))
        if candidates and cosine_distance(record.embedding, candidates[0].center) <= self.eps:
            candidates[0].members.append(record)
            self._refresh(candidates[0])
        else:
            self.noise.append(record)
            self._promote_noise()

    def evict_before(self, threshold: float) -> list[FrameRecord]:
        evicted: list[FrameRecord] = []
        self.noise, removed_noise = [item for item in self.noise if item.timestamp >= threshold], [item for item in self.noise if item.timestamp < threshold]
        evicted.extend(removed_noise)
        for cluster_id in list(self.clusters):
            cluster = self.clusters[cluster_id]
            retained = [item for item in cluster.members if item.timestamp >= threshold]
            evicted.extend(item for item in cluster.members if item.timestamp < threshold)
            cluster.members = retained
            if cluster.members:
                self._refresh(cluster)
            else:
                del self.clusters[cluster_id]
        return sorted(evicted, key=lambda item: item.timestamp)

    def retrieve(self, query: np.ndarray, top_k: int) -> list[FrameRecord]:
        clusters = sorted(self.clusters.values(), key=lambda cluster: cosine_distance(query, cluster.center))[:top_k]
        members = [member for cluster in clusters for member in cluster.members]
        if not members:
            members = sorted(self.noise, key=lambda item: cosine_distance(query, item.embedding))[:top_k]
        return sorted(members, key=lambda item: item.timestamp)


class MultiLevelMemory:
    def __init__(self, config: MyBuddyConfig, embedder: Embedder, summarizer: MLLMBackend):
        self.config = config
        self.embedder = embedder
        self.summarizer = summarizer
        self.active: deque[FrameRecord] = deque()
        self.semi = IncrementalDensityMemory(config.dbscan_eps, config.dbscan_min_pts)
        self.low_source: list[FrameRecord] = []
        self.low_summaries: list[str] = []
        self.next_summary_end = config.low_summary_window_seconds

    def ingest(self, timestamp: float, path: Path) -> None:
        self.active.append(FrameRecord(timestamp, path, self.embedder.encode_image(path)))
        self.advance(timestamp)

    def advance(self, timestamp: float) -> None:
        active_threshold = timestamp - self.config.delta_seconds
        while self.active and self.active[0].timestamp < active_threshold:
            self.semi.add(self.active.popleft())
        low_threshold = timestamp - self.config.delta_seconds - self.config.gamma_seconds
        evicted = self.semi.evict_before(low_threshold)
        if evicted:
            self.low_source.extend(evicted)
            self._summarize_ready()

    def _summarize_ready(self) -> None:
        if not self.low_source:
            return
        latest = self.low_source[-1].timestamp
        stride = self.config.low_summary_window_seconds - self.config.low_summary_overlap_seconds
        if stride <= 0:
            raise ValueError("low_summary_overlap_seconds must be smaller than low_summary_window_seconds")
        while latest >= self.next_summary_end:
            start = self.next_summary_end - self.config.low_summary_window_seconds
            records = [item for item in self.low_source if start <= item.timestamp <= self.next_summary_end]
            if records:
                selected = evenly_spaced(records, self.config.max_summary_frames)
                summary = self.summarizer.text(LOW_ACTIVE_SUMMARY, [item.path for item in selected]).strip()
                if summary:
                    self.low_summaries.append(summary)
            self.next_summary_end += stride
        keep_after = self.next_summary_end - self.config.low_summary_window_seconds
        self.low_source = [item for item in self.low_source if item.timestamp >= keep_after]

    def active_records(self) -> list[FrameRecord]:
        return list(self.active)

    def retrieve_semi(self, question: str) -> list[FrameRecord]:
        query = self.embedder.encode_text(question)
        return self.semi.retrieve(query, self.config.top_k_clusters)

