from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sac import utils


class RunLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._train_path = self.log_dir / "train.csv"
        self._eval_path = self.log_dir / "eval.csv"
        self._events_path = self.log_dir / "events.jsonl"

        self._train_file = self._train_path.open("a", newline="", encoding="utf-8")
        self._eval_file = self._eval_path.open("a", newline="", encoding="utf-8")
        self._events_file = self._events_path.open("a", encoding="utf-8")

        self._train_writer = csv.writer(self._train_file)
        self._eval_writer = csv.writer(self._eval_file)

        self._train_header = [
            "step",
            "actor_loss",
            "critic_loss",
            "alpha_loss",
            "alpha",
            "mean_q",
            "entropy",
        ]
        self._eval_header = [
            "step",
            "mean_return",
            "std_return",
            "min_return",
            "max_return",
            "n_episodes",
            "mean_episode_duration",
        ]

        if self._train_path.stat().st_size == 0:
            self._train_writer.writerow(self._train_header)
        if self._eval_path.stat().st_size == 0:
            self._eval_writer.writerow(self._eval_header)

    def log_train(self, step: int, metrics: dict[str, float]) -> None:
        row = [
            int(step),
            float(metrics["actor_loss"]),
            float(metrics["critic_loss"]),
            float(metrics["alpha_loss"]),
            float(metrics["alpha"]),
            float(metrics["mean_q"]),
            float(metrics["entropy"]),
        ]
        self._train_writer.writerow(row)

    def log_eval(self, step: int, metrics: dict[str, float]) -> None:
        row = [
            int(step),
            float(metrics["mean_return"]),
            float(metrics["std_return"]),
            float(metrics["min_return"]),
            float(metrics["max_return"]),
            float(metrics["n_episodes"]),
            float(metrics["mean_episode_duration"]),
        ]
        self._eval_writer.writerow(row)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        event = {"timestamp": utils.get_timestamp(), "event": event_type}
        event.update(payload)
        self._events_file.write(json.dumps(event) + "\n")

    def flush(self) -> None:
        self._train_file.flush()
        self._eval_file.flush()
        self._events_file.flush()

    def close(self) -> None:
        self.flush()
        self._train_file.close()
        self._eval_file.close()
        self._events_file.close()
