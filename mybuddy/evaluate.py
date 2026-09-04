import argparse
import json
from pathlib import Path

from .config import load_config
from .llm import build_backend
from .prompts import evaluation_prompt
from .utils import read_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    judge = build_backend(config.judge_backend, config.image_max_side)
    rows = read_jsonl(args.predictions)
    evaluated: list[dict] = []
    for row in rows:
        result = judge.mapping(evaluation_prompt(row["question"], row.get("answers", []), row["prediction"]))
        correct = str(result.get("pred", "no")).lower() == "yes"
        score = max(0, min(5, int(result.get("score", 0))))
        evaluated.append({**row, "correct": correct, "score": score})
    accuracy = 100.0 * sum(item["correct"] for item in evaluated) / max(1, len(evaluated))
    mean_score = sum(item["score"] for item in evaluated) / max(1, len(evaluated))
    categories: dict[str, list[dict]] = {}
    for item in evaluated:
        category = str(item.get("metadata", {}).get("category", "all"))
        categories.setdefault(category, []).append(item)
    per_category = {key: {"count": len(values), "accuracy": 100.0 * sum(item["correct"] for item in values) / len(values), "mean_score": sum(item["score"] for item in values) / len(values)} for key, values in categories.items()}
    attributes: dict[str, dict[str, list[dict]]] = {"chain": {}, "deictic": {}, "duration": {}}
    for item in evaluated:
        metadata = item.get("metadata", {})
        for name in ("chain", "deictic"):
            if name in metadata:
                key = str(bool(metadata[name])).lower()
                attributes[name].setdefault(key, []).append(item)
        timestamp = float(item.get("timestamp", 0.0))
        duration_key = "0-60" if timestamp < 60 else "60-120" if timestamp < 120 else "120-180" if timestamp < 180 else "180+"
        attributes["duration"].setdefault(duration_key, []).append(item)
    breakdowns = {name: {key: {"count": len(values), "accuracy": 100.0 * sum(item["correct"] for item in values) / len(values), "mean_score": sum(item["score"] for item in values) / len(values)} for key, values in groups.items()} for name, groups in attributes.items()}
    report = {"count": len(evaluated), "accuracy": accuracy, "mean_score": mean_score, "per_category": per_category, "breakdowns": breakdowns, "items": evaluated}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
