from .llm import MLLMBackend
from .prompts import question_filter_prompt
from .schemas import QAPair


class HistoricalQABuffer:
    def __init__(self, backend: MLLMBackend, short_window: int):
        self.backend = backend
        self.short_window = short_window
        self.items: list[QAPair] = []

    def append(self, pair: QAPair) -> None:
        self.items.append(pair)

    def _payload(self, candidates: list[QAPair]) -> list[dict]:
        return [{"qid": item.qid, "question": item.question, "answer": item.answer, "timestamp": item.timestamp} for item in candidates]

    def _search(self, question: str, candidates: list[QAPair], current_frame) -> QAPair | None:
        if not candidates:
            return None
        result = self.backend.mapping(question_filter_prompt(question, self._payload(candidates)), [current_frame] if current_frame else None)
        if str(result.get("type", "")).lower() != "chained":
            return None
        selected = str(result.get("base_qa", result.get("base qa", "")))
        return next((item for item in candidates if item.qid == selected), None)

    def retrieve(self, question: str, current_frame) -> QAPair | None:
        split = max(0, len(self.items) - self.short_window)
        short = self.items[split:]
        long = self.items[:split]
        return self._search(question, short, current_frame) or self._search(question, long, current_frame)

