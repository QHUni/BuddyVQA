import argparse
from dataclasses import asdict
from itertools import groupby
from pathlib import Path

from .config import load_config
from .data import load_samples
from .model import MyBuddy
from .utils import seed_everything, write_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=".cache/mybuddy")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    config = load_config(args.config)
    seed_everything(config.seed)
    samples = load_samples(args.data)
    model = MyBuddy(config, args.cache_dir)
    rows: list[dict] = []
    for _, group in groupby(samples, key=lambda item: item.video_id):
        rows.extend(asdict(item) for item in model.run_video(list(group)))
    write_jsonl(Path(args.output), rows)


if __name__ == "__main__":
    main()

