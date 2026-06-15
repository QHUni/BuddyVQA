from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from .config import MyBuddyConfig
from .data import QARecord, VideoExample
from .memory import MultiLevelMemory
from .models import AnswerModel, FrameEncoder, HierarchicalRephraser
from .retrieval import RetrievalResult, VisualInfoRetriever
from .router import HistoricalQABuffer, QuestionRouter, RouterPrediction


@dataclass
class QAOutput:

    qid: str
    timestamp: int
    original_question: str
    rephrased_question: str
    answer: str
    router: Dict[str, object]
    retrieval: Dict[str, object]
    memory_state: Dict[str, object]
    gold_answers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "qid": self.qid,
            "timestamp": self.timestamp,
            "original_question": self.original_question,
            "rephrased_question": self.rephrased_question,
            "answer": self.answer,
            "router": self.router,
            "retrieval": self.retrieval,
            "memory_state": self.memory_state,
            "gold_answers": self.gold_answers,
        }


class MyBuddyPipeline:

    def __init__(self, cfg: MyBuddyConfig, router: Optional[QuestionRouter] = None):
        self.cfg = cfg
        self.encoder = FrameEncoder(cfg.embedding_dim)
        self.memory = MultiLevelMemory(cfg, self.encoder)
        self.router = router or QuestionRouter(cfg.embedding_dim, cfg.router_threshold)
        self.history = HistoricalQABuffer(cfg.max_history)
        self.retriever = VisualInfoRetriever(cfg.embedding_dim)
        self.rephraser = HierarchicalRephraser()
        self.answer_model = AnswerModel(cfg.answer_max_words)
        self.current_video: Optional[VideoExample] = None

    def reset(self) -> None:
        self.memory.reset()
        self.history.clear()
        self.current_video = None

    def start_video(self, video: VideoExample) -> None:
        self.reset()
        self.current_video = video.sort()

    def answer_qa(self, qa: QARecord, video: Optional[VideoExample] = None, use_gold_history: bool = False) -> QAOutput:

        if video is not None and video is not self.current_video:
            self.start_video(video)
        if self.current_video is None:
            raise RuntimeError("No video stream has been started.")
        self.memory.update_until(self.current_video.frames, qa.timestamp)
        retrieval = self.retriever.retrieve(qa.question, self.memory)
        router_pred = self.router.predict(qa.question, self.history, qa.timestamp)
        context = self._build_context(retrieval, router_pred)
        rephrased = self.rephraser.rephrase(qa.question, context)
        answer = self.answer_model.answer(rephrased, context)
        output = QAOutput(
            qid=qa.qid,
            timestamp=qa.timestamp,
            original_question=qa.question,
            rephrased_question=rephrased,
            answer=answer,
            router=router_pred.to_dict(),
            retrieval=retrieval.to_context(),
            memory_state=self.memory.describe_state(),
            gold_answers=list(qa.answers),
        )
        stored_answer = qa.primary_answer() if use_gold_history else answer
        self.history.add(qa, stored_answer)
        return output

    def _build_context(self, retrieval: RetrievalResult, router_pred: RouterPrediction) -> Dict[str, object]:
        context = retrieval.to_context()
        context["history"] = self.history.to_context_list()
        if router_pred.related_item is not None:
            item = router_pred.related_item
            context["related_qa"] = {
                "qid": item.qid,
                "timestamp": item.timestamp,
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
            }
        else:
            context["related_qa"] = None
        return context

    def run_video(self, video: VideoExample, use_gold_history: bool = False) -> List[QAOutput]:
        self.start_video(video)
        outputs: List[QAOutput] = []
        for qa in video.qas:
            outputs.append(self.answer_qa(qa, use_gold_history=use_gold_history))
        return outputs

    def run_dataset(self, videos: Sequence[VideoExample], use_gold_history: bool = False) -> List[QAOutput]:
        all_outputs: List[QAOutput] = []
        for video in videos:
            all_outputs.extend(self.run_video(video, use_gold_history=use_gold_history))
        return all_outputs

    def interactive_answer(self, question: str, timestamp: int) -> QAOutput:
        if self.current_video is None:
            raise RuntimeError("Call start_video before interactive_answer.")
        qa = QARecord(
            video_id=self.current_video.video_id,
            qid=f"interactive_{timestamp}_{len(self.history)}",
            timestamp=timestamp,
            question=question,
            answers=[],
            category="Interactive",
        )
        return self.answer_qa(qa)

    def save_outputs_jsonl(self, outputs: Sequence[QAOutput], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def render_output(self, output: QAOutput) -> str:
        lines = [
            f"QID: {output.qid} @ {output.timestamp}s",
            f"Question: {output.original_question}",
            f"Rephrased: {output.rephrased_question}",
            f"Answer: {output.answer}",
            f"Router: {output.router}",
            f"Memory: {output.memory_state}",
        ]
        return "\n".join(lines)


def outputs_to_table(outputs: Sequence[QAOutput]) -> str:
    rows = ["| qid | question | rephrased | answer |", "|---|---|---|---|"]
    for out in outputs:
        q = out.original_question.replace("|", "/")
        r = out.rephrased_question.replace("|", "/")
        a = out.answer.replace("|", "/")
        rows.append(f"| {out.qid} | {q} | {r} | {a} |")
    return "\n".join(rows)


def run_single_demo(cfg: MyBuddyConfig, router: Optional[QuestionRouter] = None) -> List[QAOutput]:
    from .data import generate_synthetic_dataset

    video = generate_synthetic_dataset(num_videos=1, seed=cfg.seed, qas_per_video=6)[0]
    pipe = MyBuddyPipeline(cfg, router=router)
    return pipe.run_video(video, use_gold_history=False)
