"""Visual Information Retrieval for MyBuddy.

VIR retrieves visual descriptions from the multi-level memory only when a
question arrives. It also filters semi-active and low-active memories, and adds
eye-gaze / hand-box cues to help resolve ego-deictic expressions. This mirrors
Section 4.2 of the uploaded BuddyVQA paper while remaining executable locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import contains_deictic
from .memory import LowActiveSummary, MemoryCluster, MultiLevelMemory
from .models import CaptionModel, EmbodiedCueModel, EncodedFrame, TextVectorizer, cosine


@dataclass
class RetrievalResult:
    """Output of Visual Information Retrieval."""

    active_descriptions: List[str] = field(default_factory=list)
    semi_active_descriptions: List[str] = field(default_factory=list)
    low_active_summaries: List[str] = field(default_factory=list)
    embodied_cues: Dict[str, object] = field(default_factory=dict)
    selected_active_times: List[int] = field(default_factory=list)
    selected_cluster_ranges: List[Tuple[int, int]] = field(default_factory=list)
    selected_summary_ranges: List[Tuple[int, int]] = field(default_factory=list)

    def visual_descriptions(self) -> List[str]:
        return self.active_descriptions + self.semi_active_descriptions + self.low_active_summaries

    def to_context(self) -> Dict[str, object]:
        return {
            "visual_descriptions": self.visual_descriptions(),
            "embodied_cues": self.embodied_cues,
            "selected_active_times": self.selected_active_times,
            "selected_cluster_ranges": self.selected_cluster_ranges,
            "selected_summary_ranges": self.selected_summary_ranges,
        }


class VisualInfoRetriever:
    """On-demand retrieval over Active, Semi-Active and Low-Active memory."""

    def __init__(self, embedding_dim: int = 64, top_active: int = 8, top_clusters: int = 4, top_summaries: int = 4):
        self.vectorizer = TextVectorizer(embedding_dim)
        self.caption_model = CaptionModel()
        self.embodied_model = EmbodiedCueModel()
        self.top_active = top_active
        self.top_clusters = top_clusters
        self.top_summaries = top_summaries

    def retrieve(self, question: str, memory: MultiLevelMemory) -> RetrievalResult:
        """Retrieve relevant visual and embodied evidence for a question."""
        active = self._select_active(question, memory.get_active())
        clusters = self._select_clusters(question, memory.get_semi_active())
        summaries = self._select_summaries(question, memory.get_low_active())
        active_desc = self.caption_model.describe_frames(active, question)
        cluster_desc = [c.description or self.caption_model.describe_cluster(c.frames, question) for c in clusters]
        summary_desc = [s.text for s in summaries]
        cues = self.embodied_model.detect(memory.get_active(), question)
        if contains_deictic(question) and not cues.get("focus_object") and active:
            cues["focus_object"] = active[-1].frame.objects[0] if active[-1].frame.objects else None
        return RetrievalResult(
            active_descriptions=active_desc,
            semi_active_descriptions=cluster_desc,
            low_active_summaries=summary_desc,
            embodied_cues=cues,
            selected_active_times=[f.timestamp for f in active],
            selected_cluster_ranges=[(c.start, c.end) for c in clusters],
            selected_summary_ranges=[(s.start, s.end) for s in summaries],
        )

    def _select_active(self, question: str, frames: Sequence[EncodedFrame]) -> List[EncodedFrame]:
        if not frames:
            return []
        q_vec = self.vectorizer.encode(question)
        scored = []
        for rank, frame in enumerate(frames):
            sim = cosine(q_vec, frame.embedding)
            recency = (rank + 1) / len(frames)
            score = 0.65 * sim + 0.35 * recency
            if contains_deictic(question):
                score += 0.1 * recency
            scored.append((score, frame))
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        selected = [f for _, f in scored[: self.top_active]]
        selected.sort(key=lambda f: f.timestamp)
        return selected

    def _select_clusters(self, question: str, clusters: Sequence[MemoryCluster]) -> List[MemoryCluster]:
        if not clusters:
            return []
        q_vec = self.vectorizer.encode(question)
        scored = []
        for cluster in clusters:
            desc_vec = self.vectorizer.encode(cluster.description)
            score = 0.7 * cosine(q_vec, desc_vec) + 0.3 * cosine(q_vec, cluster.centroid)
            scored.append((score, cluster))
        scored.sort(key=lambda x: (x[0], x[1].end), reverse=True)
        selected = [c for _, c in scored[: self.top_clusters]]
        selected.sort(key=lambda c: c.start)
        return selected

    def _select_summaries(self, question: str, summaries: Sequence[LowActiveSummary]) -> List[LowActiveSummary]:
        if not summaries:
            return []
        q_vec = self.vectorizer.encode(question)
        scored = []
        for idx, summary in enumerate(summaries):
            sim = cosine(q_vec, self.vectorizer.encode(summary.text))
            recency = (idx + 1) / len(summaries)
            scored.append((0.75 * sim + 0.25 * recency, summary))
        scored.sort(key=lambda x: (x[0], x[1].end), reverse=True)
        selected = [s for _, s in scored[: self.top_summaries]]
        selected.sort(key=lambda s: s.start)
        return selected

    def render_debug(self, result: RetrievalResult) -> str:
        """Pretty-print retrieval output for debugging."""
        lines = ["Visual Information Retrieval"]
        lines.append(f"Active times: {result.selected_active_times}")
        lines.append(f"Semi-active ranges: {result.selected_cluster_ranges}")
        lines.append(f"Low-active ranges: {result.selected_summary_ranges}")
        lines.append(f"Embodied cues: {result.embodied_cues}")
        for desc in result.visual_descriptions()[:8]:
            lines.append(f"- {desc}")
        return "\n".join(lines)


def retrieval_precision_proxy(question: str, result: RetrievalResult) -> float:
    """A small diagnostic score measuring lexical overlap with retrieved text."""
    q_tokens = {t for t in question.lower().split() if len(t) > 2}
    if not q_tokens:
        return 0.0
    text = " ".join(result.visual_descriptions()).lower()
    hit = sum(1 for token in q_tokens if token in text)
    return hit / len(q_tokens)
