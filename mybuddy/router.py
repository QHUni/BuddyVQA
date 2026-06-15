from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import QARecord, contains_deictic, tokenize
from .models import TextVectorizer, cosine


@dataclass
class HistoryItem:

    qid: str
    timestamp: int
    question: str
    answer: str
    category: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "qid": self.qid,
            "timestamp": self.timestamp,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
        }


class HistoricalQABuffer:

    def __init__(self, max_items: int = 32):
        self.max_items = max_items
        self.items: List[HistoryItem] = []

    def add(self, qa: QARecord, answer: str) -> None:
        item = HistoryItem(qa.qid, qa.timestamp, qa.question, answer, qa.category)
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]

    def clear(self) -> None:
        self.items.clear()

    def by_id(self, qid: str) -> Optional[HistoryItem]:
        for item in self.items:
            if item.qid == qid:
                return item
        return None

    def latest(self) -> Optional[HistoryItem]:
        return self.items[-1] if self.items else None

    def to_context_list(self) -> List[Dict[str, object]]:
        return [item.to_dict() for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


@dataclass
class RouterPrediction:

    is_chained: bool
    probability: float
    related_qid: Optional[str]
    related_item: Optional[HistoryItem]
    debug_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "is_chained": self.is_chained,
            "probability": self.probability,
            "related_qid": self.related_qid,
            "debug_scores": self.debug_scores,
        }


class QuestionRouter:

    def __init__(self, dim: int = 64, threshold: float = 0.52):
        self.vectorizer = TextVectorizer(dim)
        self.threshold = threshold
        self.weights = [0.8, 1.2, 0.6, -0.4, 0.4, 0.2]
        self.bias = -0.55

    def features(self, question: str, history_item: Optional[HistoryItem], current_time: int) -> List[float]:
        deictic = 1.0 if contains_deictic(question) else 0.0
        if history_item is None:
            return [deictic, 0.0, 0.0, 0.0, 0.0, 1.0]
        q_vec = self.vectorizer.encode(question)
        h_vec = self.vectorizer.encode(history_item.question + " " + history_item.answer)
        sim = cosine(q_vec, h_vec)
        gap = max(0, current_time - history_item.timestamp)
        recency = math.exp(-gap / 90.0)
        same_category_hint = 1.0 if any(tok in tokenize(history_item.question) for tok in tokenize(question)) else 0.0
        history_len_proxy = min(1.0, len(tokenize(history_item.question + " " + history_item.answer)) / 40.0)
        return [deictic, sim, recency, gap / 180.0, same_category_hint, history_len_proxy]

    def predict(self, question: str, history: HistoricalQABuffer, current_time: int) -> RouterPrediction:
        if len(history) == 0:
            prob = self._sigmoid(self._score(self.features(question, None, current_time)))
            return RouterPrediction(False, prob, None, None, {})
        debug: Dict[str, float] = {}
        best_item: Optional[HistoryItem] = None
        best_prob = -1.0
        for item in history:
            feats = self.features(question, item, current_time)
            prob = self._sigmoid(self._score(feats))
            debug[item.qid] = prob
            if prob > best_prob:
                best_prob = prob
                best_item = item
        is_chained = best_prob >= self.threshold
        return RouterPrediction(is_chained, best_prob, best_item.qid if is_chained and best_item else None, best_item if is_chained else None, debug)

    def train(self, examples: Sequence[Tuple[str, int, List[HistoryItem], Optional[str]]], epochs: int = 10, lr: float = 0.1) -> Dict[str, float]:
        pairs: List[Tuple[List[float], float]] = []
        for question, ts, candidates, gold in examples:
            if not candidates:
                pairs.append((self.features(question, None, ts), 0.0 if gold is None else 1.0))
                continue
            for item in candidates:
                label = 1.0 if gold is not None and item.qid == gold else 0.0
                pairs.append((self.features(question, item, ts), label))
        if not pairs:
            return {"loss": 0.0, "pairs": 0}
        for _ in range(epochs):
            total_loss = 0.0
            grad_w = [0.0 for _ in self.weights]
            grad_b = 0.0
            for feats, label in pairs:
                pred = self._sigmoid(self._score(feats))
                pred = min(max(pred, 1e-6), 1 - 1e-6)
                total_loss += -(label * math.log(pred) + (1 - label) * math.log(1 - pred))
                diff = pred - label
                for i, value in enumerate(feats):
                    grad_w[i] += diff * value
                grad_b += diff
            scale = 1.0 / len(pairs)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * grad_w[i] * scale
            self.bias -= lr * grad_b * scale
        return {"loss": self.loss(pairs), "pairs": float(len(pairs))}

    def loss(self, pairs: Sequence[Tuple[List[float], float]]) -> float:
        if not pairs:
            return 0.0
        total = 0.0
        for feats, label in pairs:
            pred = min(max(self._sigmoid(self._score(feats)), 1e-6), 1 - 1e-6)
            total += -(label * math.log(pred) + (1 - label) * math.log(1 - pred))
        return total / len(pairs)

    def _score(self, feats: Sequence[float]) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, feats))

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights, "bias": self.bias, "threshold": self.threshold, "dim": self.vectorizer.dim}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "QuestionRouter":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        router = cls(dim=int(data.get("dim", 64)), threshold=float(data.get("threshold", 0.52)))
        router.weights = [float(x) for x in data.get("weights", router.weights)]
        router.bias = float(data.get("bias", router.bias))
        return router


def build_router_training_examples(videos: Sequence[object]) -> List[Tuple[str, int, List[HistoryItem], Optional[str]]]:
    examples: List[Tuple[str, int, List[HistoryItem], Optional[str]]] = []
    for video in videos:
        buffer = HistoricalQABuffer(max_items=64)
        for qa in video.qas:
            examples.append((qa.question, qa.timestamp, list(buffer.items), qa.chained_to))
            buffer.add(qa, qa.primary_answer())
    return examples


def router_accuracy(router: QuestionRouter, videos: Sequence[object]) -> Dict[str, float]:
    total = 0
    cls_ok = 0
    rel_total = 0
    rel_ok = 0
    for video in videos:
        buffer = HistoricalQABuffer(max_items=64)
        for qa in video.qas:
            pred = router.predict(qa.question, buffer, qa.timestamp)
            gold_chained = qa.chained_to is not None
            cls_ok += int(pred.is_chained == gold_chained)
            total += 1
            if gold_chained:
                rel_total += 1
                rel_ok += int(pred.related_qid == qa.chained_to)
            buffer.add(qa, qa.primary_answer())
    return {
        "router_cls_acc": cls_ok / max(1, total),
        "router_rel_acc": rel_ok / max(1, rel_total),
        "router_total": float(total),
    }
