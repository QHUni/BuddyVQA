"""Configuration utilities for the MyBuddy project.

The real paper describes a train-free framework, yet the user requested a full
training and inference code project. This module keeps configuration compact and
explicit so both the light router training and the streaming inference pipeline
share exactly the same settings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class MyBuddyConfig:
    """Runtime configuration shared by all modules.

    The names follow the paper notation: active_window corresponds to Delta,
    semi_active_window corresponds to Gamma, and dbscan_eps/min_pts correspond
    to the density-based compression hyper-parameters used in semi-active
    memory. The remaining options are engineering parameters used by this
    runnable implementation.
    """

    sample_fps: int = 1
    active_window: int = 30
    semi_active_window: int = 120
    dbscan_eps: float = 0.27
    dbscan_min_pts: int = 5
    low_active_summary_window: int = 60
    max_history: int = 32
    embedding_dim: int = 64
    seed: int = 42
    router_threshold: float = 0.52
    answer_max_words: int = 48
    train_size: int = 120
    val_size: int = 48
    test_size: int = 48
    output_dir: str = "runs/demo"
    dataset_path: Optional[str] = None
    checkpoint: Optional[str] = None
    verbose: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values and raise a clear error if invalid."""
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.active_window <= 0:
            raise ValueError("active_window must be positive")
        if self.semi_active_window <= self.active_window:
            raise ValueError("semi_active_window should be larger than active_window")
        if self.dbscan_eps <= 0:
            raise ValueError("dbscan_eps must be positive")
        if self.dbscan_min_pts <= 0:
            raise ValueError("dbscan_min_pts must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= self.router_threshold <= 1.0:
            raise ValueError("router_threshold must be in [0, 1]")
        if self.answer_max_words <= 0:
            raise ValueError("answer_max_words must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyBuddyConfig":
        """Create a config from a dictionary while keeping unknown keys."""
        known = {k for k in cls.__dataclass_fields__.keys()}
        kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known and key != "extras":
                kwargs[key] = value
            else:
                extras[key] = value
        cfg = cls(**kwargs)
        cfg.extras.update(extras)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str) -> "MyBuddyConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def update_from_args(self, args: argparse.Namespace) -> "MyBuddyConfig":
        """Return a new config updated with non-None argparse values."""
        data = self.to_dict()
        for key, value in vars(args).items():
            if value is not None and key in data:
                data[key] = value
        return MyBuddyConfig.from_dict(data)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach common command-line options to an argparse parser."""
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional BuddyVQA-style JSONL file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for logs and checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a router checkpoint JSON.")
    parser.add_argument("--sample_fps", type=int, default=None)
    parser.add_argument("--active_window", type=int, default=None)
    parser.add_argument("--semi_active_window", type=int, default=None)
    parser.add_argument("--dbscan_eps", type=float, default=None)
    parser.add_argument("--dbscan_min_pts", type=int, default=None)
    parser.add_argument("--low_active_summary_window", type=int, default=None)
    parser.add_argument("--max_history", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--router_threshold", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print detailed pipeline states.")
    return parser


def load_config_from_args(args: argparse.Namespace) -> MyBuddyConfig:
    """Load config from CLI arguments.

    A config file is optional. When supplied, it forms the base configuration and
    explicit CLI flags override it. This mirrors common research-code practice.
    """
    if args.config:
        cfg = MyBuddyConfig.from_json(args.config)
    else:
        cfg = MyBuddyConfig()
    cfg = cfg.update_from_args(args)
    if getattr(args, "verbose", False):
        cfg.verbose = True
    cfg.validate()
    return cfg


def seed_everything(seed: int) -> None:
    """Seed Python's random module and common hash-dependent behavior."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def merge_dicts(base: Dict[str, Any], updates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Shallow merge dictionaries from left to right."""
    merged = dict(base)
    for item in updates:
        for key, value in item.items():
            if value is not None:
                merged[key] = value
    return merged
