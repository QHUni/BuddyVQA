"""Evaluation utilities and CLI for MyBuddy.

The paper evaluates open-ended answers with an LLM judge. This runnable project
cannot assume access to GPT-style APIs, so it provides a deterministic lexical
judge and router diagnostics. The output is still organized around the same
research axes: overall accuracy, category accuracy, chained question accuracy,
deictic-question accuracy, router accuracy, and simple confidence proxies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import add_common_args, load_config_from_args, seed_everything
from .data import QARecord, VideoExample, generate_synthetic_dataset, load_jsonl, tokenize
from .pipeline import MyBuddyPipeline, QAOutput
from .router import QuestionRouter, router_accuracy


@dataclass
class JudgeResult:
    """Result produced by the local answer judge."""

    correct: bool
    score: float
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {"correct": self.correct, "score": self.score, "confidence": self.confidence, "reason": self.reason}


class LexicalAnswerJudge:
    """A deterministic substitute for an LLM answer judge."""

    def __init__(self, min_overlap: float = 0.22):
        self.min_overlap = min_overlap
        self.stopwords = {
            "the", "a", "an", "is", "are", "to", "of", "and", "or", "it", "in", "on", "at", "near", "with", "you", "your",
            "based", "previous", "answer", "appears", "seems", "likely", "current", "visible",
        }

    def judge(self, prediction: str, gold_answers: Sequence[str]) -> JudgeResult:
        if not gold_answers:
            return JudgeResult(False, 0.0, 0.0, "No gold answers were provided.")
        best = 0.0
        best_gold = ""
        for gold in gold_answers:
            overlap = self._overlap(prediction, gold)
            if overlap > best:
                best = overlap
                best_gold = gold
        correct = best >= self.min_overlap or self._substring_match(prediction, best_gold)
        score = min(5.0, round(best * 5.0, 2))
        confidence = min(5.0, round((best + (0.2 if correct else 0.0)) * 5.0, 2))
        reason = f"Best lexical overlap={best:.3f} with gold='{best_gold}'."
        return JudgeResult(correct, score, confidence, reason)

    def _overlap(self, a: str, b: str) -> float:
        ta = {t for t in tokenize(a) if t and t not in self.stopwords and len(t) > 2}
        tb = {t for t in tokenize(b) if t and t not in self.stopwords and len(t) > 2}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(tb))

    def _substring_match(self, a: str, b: str) -> bool:
        aa = re.sub(r"\s+", " ", a.lower()).strip()
        bb = re.sub(r"\s+", " ", b.lower()).strip()
        return bool(aa and bb and (aa in bb or bb in aa))


def load_eval_videos(cfg, num_videos: int, qas_per_video: int) -> List[VideoExample]:
    if cfg.dataset_path:
        return load_jsonl(cfg.dataset_path)
    return generate_synthetic_dataset(num_videos=num_videos, seed=cfg.seed + 202, qas_per_video=qas_per_video)


def maybe_load_router(cfg) -> QuestionRouter:
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        return QuestionRouter.load(cfg.checkpoint)
    return QuestionRouter(cfg.embedding_dim, cfg.router_threshold)


def aggregate(outputs: Sequence[QAOutput], judgments: Sequence[JudgeResult], videos: Sequence[VideoExample]) -> Dict[str, float]:
    """Aggregate answer accuracy by category, chained status and deictic status."""
    qa_map: Dict[str, QARecord] = {qa.qid: qa for video in videos for qa in video.qas}
    buckets: Dict[str, List[int]] = {"overall": []}
    scores: List[float] = []
    confidence: List[float] = []
    for out, judge in zip(outputs, judgments):
        qa = qa_map.get(out.qid)
        value = int(judge.correct)
        buckets.setdefault("overall", []).append(value)
        scores.append(judge.score)
        confidence.append(judge.confidence)
        if qa is None:
            continue
        buckets.setdefault(f"category_{qa.category}", []).append(value)
        buckets.setdefault("chained" if qa.is_chained else "unchained", []).append(value)
        buckets.setdefault("deictic" if qa.has_deictic else "non_deictic", []).append(value)
    metrics: Dict[str, float] = {}
    for key, values in buckets.items():
        metrics[f"acc_{key}"] = sum(values) / max(1, len(values))
        metrics[f"count_{key}"] = float(len(values))
    metrics["avg_score"] = sum(scores) / max(1, len(scores))
    metrics["avg_confidence"] = sum(confidence) / max(1, len(confidence))
    return metrics


def evaluate_pipeline(cfg, videos: Sequence[VideoExample], router: QuestionRouter, use_gold_history: bool) -> Tuple[List[QAOutput], List[JudgeResult], Dict[str, float]]:
    pipeline = MyBuddyPipeline(cfg, router=router)
    outputs = pipeline.run_dataset(videos, use_gold_history=use_gold_history)
    judge = LexicalAnswerJudge()
    judgments = [judge.judge(out.answer, out.gold_answers) for out in outputs]
    metrics = aggregate(outputs, judgments, videos)
    metrics.update(router_accuracy(router, videos))
    return outputs, judgments, metrics


def save_eval_report(path: str, outputs: Sequence[QAOutput], judgments: Sequence[JudgeResult], metrics: Dict[str, float]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"metrics": metrics}, ensure_ascii=False) + "\n")
        for out, judge in zip(outputs, judgments):
            row = out.to_dict()
            row["judge"] = judge.to_dict()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MyBuddy on BuddyVQA-style data.")
    add_common_args(parser)
    parser.add_argument("--num_videos", type=int, default=16)
    parser.add_argument("--qas_per_video", type=int, default=8)
    parser.add_argument("--report", type=str, default=None, help="Optional JSONL report path.")
    parser.add_argument("--use_gold_history", action="store_true", help="Use gold previous answers for history updates.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_config_from_args(args)
    seed_everything(cfg.seed)
    videos = load_eval_videos(cfg, args.num_videos, args.qas_per_video)
    router = maybe_load_router(cfg)
    outputs, judgments, metrics = evaluate_pipeline(cfg, videos, router, args.use_gold_history)
    if args.report:
        save_eval_report(args.report, outputs, judgments, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
