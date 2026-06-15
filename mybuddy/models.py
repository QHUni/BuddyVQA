from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import DEICTIC_WORDS, FrameRecord, QARecord, contains_deictic, tokenize

Vector = List[float]


def stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def l2_normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [float(x) / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class TextVectorizer:

    def __init__(self, dim: int = 64):
        self.dim = int(dim)

    def encode(self, text: str) -> Vector:
        vec = [0.0 for _ in range(self.dim)]
        for token in tokenize(text):
            h = stable_hash(token)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return l2_normalize(vec)

    def encode_many(self, texts: Iterable[str]) -> List[Vector]:
        return [self.encode(t) for t in texts]


@dataclass
class EncodedFrame:

    frame: FrameRecord
    embedding: Vector
    caption: str

    @property
    def timestamp(self) -> int:
        return self.frame.timestamp

    def short_text(self) -> str:
        objs = ", ".join(self.frame.objects) if self.frame.objects else "objects"
        return f"t={self.timestamp}s: {objs} near {self.frame.location}"


class FrameEncoder:

    def __init__(self, dim: int = 64):
        self.vectorizer = TextVectorizer(dim)

    def encode_frame(self, frame: FrameRecord) -> EncodedFrame:
        text = " ".join([frame.text, frame.location, " ".join(frame.objects)])
        emb = self.vectorizer.encode(text)
        caption = self.caption_frame(frame)
        return EncodedFrame(frame=frame, embedding=emb, caption=caption)

    def encode_stream(self, frames: Sequence[FrameRecord]) -> List[EncodedFrame]:
        return [self.encode_frame(f) for f in frames]

    def caption_frame(self, frame: FrameRecord) -> str:
        objects = ", ".join(frame.objects) if frame.objects else "unknown objects"
        return f"At {frame.timestamp}s, the wearer sees {objects} around the {frame.location}. {frame.text}"


class CaptionModel:

    def describe_frames(self, frames: Sequence[EncodedFrame], question: str = "") -> List[str]:
        descriptions: List[str] = []
        for enc in frames:
            focus = self._focus_suffix(enc, question)
            descriptions.append(f"{enc.caption}{focus}")
        return descriptions

    def describe_cluster(self, frames: Sequence[EncodedFrame], question: str = "") -> str:
        if not frames:
            return "No mid-term visual event is available."
        start, end = frames[0].timestamp, frames[-1].timestamp
        object_counts: Dict[str, int] = {}
        locations: Dict[str, int] = {}
        for enc in frames:
            for obj in enc.frame.objects:
                object_counts[obj] = object_counts.get(obj, 0) + 1
            loc = enc.frame.location
            locations[loc] = locations.get(loc, 0) + 1
        top_objects = sorted(object_counts, key=object_counts.get, reverse=True)[:3]
        top_locations = sorted(locations, key=locations.get, reverse=True)[:2]
        obj_text = ", ".join(top_objects) if top_objects else "objects"
        loc_text = ", ".join(top_locations) if top_locations else "places"
        return f"From {start}s to {end}s, recurring visual content includes {obj_text} near {loc_text}."

    def _focus_suffix(self, enc: EncodedFrame, question: str) -> str:
        q_tokens = set(tokenize(question))
        overlap = [o for o in enc.frame.objects if o.lower() in q_tokens]
        if overlap:
            return f" The question may focus on {', '.join(overlap)}."
        if contains_deictic(question) and enc.frame.objects:
            return f" The deictic reference may point to {enc.frame.objects[0]}."
        return ""


class SummaryModel:

    def summarize(self, descriptions: Sequence[str], start: int, end: int) -> str:
        if not descriptions:
            return f"Summary {start}-{end}s: no visible evidence."
        words: Dict[str, int] = {}
        for desc in descriptions:
            for tok in tokenize(desc):
                if len(tok) <= 2:
                    continue
                words[tok] = words.get(tok, 0) + 1
        keywords = sorted(words, key=words.get, reverse=True)[:8]
        key_text = ", ".join(keywords) if keywords else "general activity"
        first = descriptions[0].split(".")[0]
        last = descriptions[-1].split(".")[0]
        return f"Summary {start}-{end}s: {first}; later {last}. Key clues: {key_text}."


class EmbodiedCueModel:

    def detect(self, frames: Sequence[EncodedFrame], question: str) -> Dict[str, object]:
        if not frames:
            return {"gaze": (0.5, 0.5), "hand_bbox": (0.35, 0.35, 0.65, 0.65), "focus_object": None}
        latest = frames[-1].frame
        focus = self._infer_focus_object(latest, question)
        return {
            "gaze": latest.gaze,
            "hand_bbox": latest.hand_bbox,
            "focus_object": focus,
            "evidence_timestamp": latest.timestamp,
        }

    def _infer_focus_object(self, frame: FrameRecord, question: str) -> Optional[str]:
        q_tokens = set(tokenize(question))
        for obj in frame.objects:
            if obj.lower() in q_tokens:
                return obj
        if contains_deictic(question) and frame.objects:
            return frame.objects[0]
        return frame.objects[0] if frame.objects else None


class HierarchicalRephraser:

    def rephrase(self, question: str, context: Dict[str, object]) -> str:
        if not contains_deictic(question):
            return question
        focus = self._select_referent(question, context)
        if not focus:
            return question
        rewritten = question
        replacements = {
            " it ": f" the {focus} ",
            " this ": f" the {focus} ",
            " that ": f" the {focus} ",
            " these ": f" the {focus} ",
            " those ": f" the {focus} ",
            " here ": " in the current place ",
            " there ": " in the referenced place ",
        }
        padded = f" {rewritten} "
        lower = padded.lower()
        for key, value in replacements.items():
            if key in lower:
                pattern = re.compile(re.escape(key.strip()), re.IGNORECASE)
                rewritten = pattern.sub(value.strip(), rewritten, count=1)
                break
        return rewritten

    def _select_referent(self, question: str, context: Dict[str, object]) -> Optional[str]:
        cues = context.get("embodied_cues", {}) or {}
        if isinstance(cues, dict) and cues.get("focus_object"):
            return str(cues["focus_object"])
        related = context.get("related_qa")
        if related and isinstance(related, dict):
            answer = str(related.get("answer", ""))
            for token in tokenize(answer):
                if token in {"knife", "phone", "mango", "bottle", "tent", "bag", "shoe", "bucket", "cup", "sauce"}:
                    return token
        visual = context.get("visual_descriptions", [])
        for desc in visual if isinstance(visual, list) else []:
            for token in tokenize(str(desc)):
                if token in {"knife", "phone", "mango", "bottle", "tent", "bag", "shoe", "bucket", "cup", "sauce"}:
                    return token
        return None


class AnswerModel:

    def __init__(self, max_words: int = 48):
        self.max_words = max_words

    def answer(self, question: str, context: Dict[str, object]) -> str:
        q = question.lower()
        visual = context.get("visual_descriptions", [])
        related = context.get("related_qa")
        cues = context.get("embodied_cues", {}) or {}
        focus_object = cues.get("focus_object") if isinstance(cues, dict) else None
        evidence = self._best_visual_sentence(visual, focus_object)
        if "where" in q:
            response = self._answer_where(evidence, focus_object, related)
        elif "how" in q or "advice" in q or "should" in q:
            response = self._answer_advice(evidence, focus_object, related)
        elif "safe" in q or "good" in q or "fresh" in q:
            response = self._answer_evaluation(evidence, focus_object)
        elif "what" in q:
            response = self._answer_what(evidence, focus_object)
        else:
            response = self._answer_generic(evidence, related)
        return self._truncate(response)

    def _best_visual_sentence(self, visual: object, focus_object: object) -> str:
        if isinstance(visual, list) and visual:
            if focus_object:
                for item in visual:
                    if str(focus_object).lower() in str(item).lower():
                        return str(item)
            return str(visual[0])
        return "The current view contains limited visual evidence."

    def _answer_where(self, evidence: str, focus_object: object, related: object) -> str:
        focus = str(focus_object) if focus_object else "item"
        loc = self._extract_location(evidence)
        if related and isinstance(related, dict):
            return f"Based on the related previous QA, the {focus} appears near the {loc}."
        return f"The {focus} appears near the {loc}."

    def _answer_advice(self, evidence: str, focus_object: object, related: object) -> str:
        focus = str(focus_object) if focus_object else "item"
        loc = self._extract_location(evidence)
        return f"Use the {focus} carefully, keep the area around the {loc} clear, and follow the current visual context."

    def _answer_evaluation(self, evidence: str, focus_object: object) -> str:
        focus = str(focus_object) if focus_object else "action"
        loc = self._extract_location(evidence)
        return f"It seems acceptable if you handle the {focus} carefully near the {loc}; watch for clutter or occlusion."

    def _answer_what(self, evidence: str, focus_object: object) -> str:
        objs = self._extract_objects(evidence)
        if focus_object and objs:
            others = [o for o in objs if o != str(focus_object)]
            if others:
                return f"The visible item next to the {focus_object} is likely the {others[0]}."
        return f"The visible evidence suggests: {evidence}"

    def _answer_generic(self, evidence: str, related: object) -> str:
        if related and isinstance(related, dict):
            return f"Using the previous QA context and current view, {evidence}"
        return evidence

    def _extract_location(self, evidence: str) -> str:
        locations = ["counter", "shelf", "table", "trunk", "basket", "corner", "floor", "yard", "sink", "rack"]
        low = evidence.lower()
        for loc in locations:
            if loc in low:
                return loc
        return "visible area"

    def _extract_objects(self, evidence: str) -> List[str]:
        objects = ["knife", "phone", "mango", "bottle", "tent", "bag", "shoe", "bucket", "cup", "sauce"]
        low = evidence.lower()
        return [o for o in objects if o in low]

    def _truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_words:
            return text
        return " ".join(words[: self.max_words]).rstrip(".,") + "."
