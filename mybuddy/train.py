from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

from .config import MyBuddyConfig, add_common_args, load_config_from_args, seed_everything
from .data import VideoExample, generate_synthetic_dataset, load_jsonl, save_jsonl, split_dataset
from .router import QuestionRouter, build_router_training_examples, router_accuracy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MyBuddy lightweight question router.")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=20, help="Number of router training epochs.")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for logistic router.")
    parser.add_argument("--num_videos", type=int, default=64, help="Synthetic video count if no dataset is given.")
    parser.add_argument("--qas_per_video", type=int, default=8, help="Synthetic QA turns per video.")
    parser.add_argument("--save_synthetic", action="store_true", help="Save generated synthetic dataset.")
    return parser


def load_or_generate_dataset(cfg: MyBuddyConfig, num_videos: int, qas_per_video: int) -> List[VideoExample]:
    if cfg.dataset_path:
        return load_jsonl(cfg.dataset_path)
    return generate_synthetic_dataset(num_videos=num_videos, seed=cfg.seed, qas_per_video=qas_per_video)


def summarize_dataset(videos: Sequence[VideoExample]) -> Dict[str, float]:
    total_qas = 0
    chained = 0
    deictic = 0
    categories: Dict[str, int] = {}
    for video in videos:
        for qa in video.qas:
            total_qas += 1
            chained += int(qa.is_chained)
            deictic += int(qa.has_deictic)
            categories[qa.category] = categories.get(qa.category, 0) + 1
    stats: Dict[str, float] = {
        "videos": float(len(videos)),
        "qas": float(total_qas),
        "chained_rate": chained / max(1, total_qas),
        "deictic_rate": deictic / max(1, total_qas),
    }
    for key, value in categories.items():
        stats[f"category_{key}"] = float(value)
    return stats


def train_router(cfg: MyBuddyConfig, train_videos: Sequence[VideoExample], val_videos: Sequence[VideoExample], epochs: int, lr: float) -> Tuple[QuestionRouter, Dict[str, float]]:
    router = QuestionRouter(cfg.embedding_dim, cfg.router_threshold)
    examples = build_router_training_examples(train_videos)
    train_log = router.train(examples, epochs=epochs, lr=lr)
    train_metrics = router_accuracy(router, train_videos)
    val_metrics = router_accuracy(router, val_videos)
    metrics: Dict[str, float] = {}
    metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
    metrics.update({f"opt_{k}": v for k, v in train_log.items()})
    return router, metrics


def save_training_artifacts(cfg: MyBuddyConfig, router: QuestionRouter, metrics: Dict[str, float], videos: Sequence[VideoExample], save_synthetic: bool) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.output_dir, "config.json"))
    checkpoint = os.path.join(cfg.output_dir, "router.json")
    router.save(checkpoint)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    if save_synthetic and not cfg.dataset_path:
        save_jsonl(videos, os.path.join(cfg.output_dir, "synthetic_dataset.jsonl"))
    return checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_config_from_args(args)
    seed_everything(cfg.seed)
    videos = load_or_generate_dataset(cfg, args.num_videos, args.qas_per_video)
    train_videos, val_videos, test_videos = split_dataset(videos, seed=cfg.seed)
    stats = summarize_dataset(videos)
    router, metrics = train_router(cfg, train_videos, val_videos, args.epochs, args.lr)
    test_metrics = router_accuracy(router, test_videos)
    metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
    metrics.update({f"data_{k}": v for k, v in stats.items()})
    checkpoint = save_training_artifacts(cfg, router, metrics, videos, args.save_synthetic)
    print(json.dumps({"checkpoint": checkpoint, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
