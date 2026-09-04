from pathlib import Path

from .config import MyBuddyConfig
from .embeddings import build_embedder
from .history import HistoricalQABuffer
from .llm import build_backend
from .memory import MultiLevelMemory
from .prompts import final_qa_prompt, visual_retrieval_prompt
from .schemas import Prediction, QAPair, Sample
from .utils import evenly_spaced
from .video import VideoFrameStream


class MyBuddy:
    def __init__(self, config: MyBuddyConfig, cache_root: str | Path):
        self.config = config
        self.cache_root = Path(cache_root)
        self.embedder = build_embedder(config.embedder)
        self.small_backend = build_backend(config.small_backend, config.image_max_side)
        self.main_backend = build_backend(config.main_backend, config.image_max_side)

    def run_video(self, samples: list[Sample]) -> list[Prediction]:
        if not samples:
            return []
        video_id = samples[0].video_id
        video_path = samples[0].video_path
        memory = MultiLevelMemory(self.config, self.embedder, self.small_backend)
        history = HistoricalQABuffer(self.small_backend, self.config.question_window)
        stream = iter(VideoFrameStream(video_path, self.cache_root / video_id, self.config.sampling_fps))
        next_frame = next(stream, None)
        predictions: list[Prediction] = []
        for sample in samples:
            while next_frame is not None and next_frame[0] <= sample.timestamp + 1e-6:
                memory.ingest(next_frame[0], next_frame[1])
                next_frame = next(stream, None)
            memory.advance(sample.timestamp)
            active = memory.active_records()
            semi = memory.retrieve_semi(sample.question)
            active_selected = evenly_spaced(active, self.config.max_active_frames)
            semi_selected = evenly_spaced(semi, self.config.max_semi_frames)
            current_frame = active[-1].path if active else None
            relevant_pair = history.retrieve(sample.question, current_frame)
            images = [item.path for item in active_selected + semi_selected]
            retrieval = self.small_backend.mapping(visual_retrieval_prompt(sample.question, [item.timestamp for item in active_selected], [item.timestamp for item in semi_selected], memory.low_summaries), images)
            coordinates = retrieval.get("coordinates", [])
            relevant = None if relevant_pair is None else {"qid": relevant_pair.qid, "question": relevant_pair.question, "answer": relevant_pair.answer, "timestamp": relevant_pair.timestamp}
            result = self.main_backend.mapping(final_qa_prompt(sample.question, retrieval, relevant, coordinates))
            rephrased = str(result.get("rephrased_question", result.get("rephrased question", sample.question)))
            answer = str(result.get("answer", ""))
            history.append(QAPair(sample.qid, sample.question, answer, sample.timestamp))
            predictions.append(Prediction(sample.video_id, sample.qid, sample.timestamp, sample.question, sample.answers, sample.metadata, rephrased, answer, relevant, retrieval))
        return predictions
