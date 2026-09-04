import json
import os
import re
from pathlib import Path
from typing import Protocol

from .config import BackendConfig
from .utils import image_data_url, parse_mapping


class MLLMBackend(Protocol):
    def text(self, prompt: str, images: list[Path] | None = None) -> str:
        ...

    def mapping(self, prompt: str, images: list[Path] | None = None) -> dict:
        ...


class OpenAIBackend:
    def __init__(self, config: BackendConfig, image_max_side: int):
        from openai import OpenAI

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing API key environment variable: {config.api_key_env}")
        options = {"api_key": api_key}
        if config.base_url:
            options["base_url"] = config.base_url
        self.client = OpenAI(**options)
        self.model = config.model
        self.image_max_side = image_max_side

    def _content(self, prompt: str, images: list[Path] | None) -> list[dict]:
        content = [{"type": "text", "text": prompt}]
        for path in images or []:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path, self.image_max_side)}})
        return content

    def text(self, prompt: str, images: list[Path] | None = None) -> str:
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": self._content(prompt, images)}], temperature=0)
        return response.choices[0].message.content or ""

    def mapping(self, prompt: str, images: list[Path] | None = None) -> dict:
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": self._content(prompt, images)}], temperature=0, response_format={"type": "json_object"})
        return parse_mapping(response.choices[0].message.content or "")


class MockBackend:
    def text(self, prompt: str, images: list[Path] | None = None) -> str:
        return "The user is in a stable indoor environment and interacts with nearby objects."

    def mapping(self, prompt: str, images: list[Path] | None = None) -> dict:
        if "base_qa" in prompt:
            qids = re.findall(r'"qid":\s*"([^"]+)"', prompt)
            chained = bool(re.search(r"\b(it|this|that|there|they|them|one)\b", prompt.split("Historical QAs:")[0], re.IGNORECASE)) and bool(qids)
            return {"type": "chained" if chained else "unchained", "base_qa": qids[-1] if chained else None}
        if "rephrased_question" in prompt:
            match = re.search(r"Incoming Question:\s*(.+)", prompt)
            question = match.group(1).strip() if match else ""
            return {"rephrased_question": question, "answer": "unknown"}
        if "filtered_memory" in prompt:
            match = re.search(r"Low-active summaries:\s*(.*)", prompt)
            summaries = json.loads(match.group(1)) if match else []
            return {"caption": "Visible egocentric scene with nearby objects.", "filtered_memory": summaries, "coordinates": []}
        if "keys pred and score" in prompt:
            prediction = re.search(r"Predicted Answer:\s*(.*)", prompt)
            answers = re.search(r"Correct Answer Set:\s*(.*)\n", prompt)
            pred_text = prediction.group(1).strip().lower() if prediction else ""
            answer_values = json.loads(answers.group(1)) if answers else []
            matched = any(pred_text == str(value).strip().lower() for value in answer_values)
            return {"pred": "yes" if matched else "no", "score": 5 if matched else 0}
        return {}


def build_backend(config: BackendConfig, image_max_side: int) -> MLLMBackend:
    if config.type == "openai":
        return OpenAIBackend(config, image_max_side)
    if config.type == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend type: {config.type}")
