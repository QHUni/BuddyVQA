from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

from .config import add_common_args, load_config_from_args, seed_everything
from .data import VideoExample, generate_synthetic_dataset, load_jsonl
from .pipeline import MyBuddyPipeline, outputs_to_table, run_single_demo
from .router import QuestionRouter


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MyBuddy streaming inference.")
    add_common_args(parser)
    parser.add_argument("--demo", action="store_true", help="Run a built-in synthetic demo.")
    parser.add_argument("--num_videos", type=int, default=2, help="Number of synthetic videos when no dataset is provided.")
    parser.add_argument("--qas_per_video", type=int, default=6, help="Synthetic QA turns per video.")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Where to save detailed inference outputs.")
    parser.add_argument("--markdown", action="store_true", help="Print outputs as a Markdown table.")
    parser.add_argument("--use_gold_history", action="store_true", help="Store gold answers in history after each turn.")
    return parser


def maybe_load_router(cfg) -> Optional[QuestionRouter]:
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        return QuestionRouter.load(cfg.checkpoint)
    return None


def load_inference_videos(cfg, num_videos: int, qas_per_video: int) -> List[VideoExample]:
    if cfg.dataset_path:
        return load_jsonl(cfg.dataset_path)
    return generate_synthetic_dataset(num_videos=num_videos, seed=cfg.seed + 99, qas_per_video=qas_per_video)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_config_from_args(args)
    seed_everything(cfg.seed)
    router = maybe_load_router(cfg)
    if args.demo:
        outputs = run_single_demo(cfg, router=router)
    else:
        videos = load_inference_videos(cfg, args.num_videos, args.qas_per_video)
        pipeline = MyBuddyPipeline(cfg, router=router)
        outputs = pipeline.run_dataset(videos, use_gold_history=args.use_gold_history)
    if args.output_jsonl:
        parent = os.path.dirname(args.output_jsonl)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for out in outputs:
                f.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")
    if args.markdown:
        print(outputs_to_table(outputs))
    else:
        for out in outputs:
            print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()