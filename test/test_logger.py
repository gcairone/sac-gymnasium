from __future__ import annotations

import csv
import json
from pathlib import Path

from sac import utils
from sac.logger import RunLogger


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_logger_writes_headers_and_rows(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    logger = RunLogger(log_dir)

    logger.log_train(
        10,
        {
            "actor_loss": 1.0,
            "critic_loss": 2.0,
            "alpha_loss": 3.0,
            "alpha": 0.2,
            "mean_q": 4.0,
            "entropy": 5.0,
        },
    )
    logger.log_eval(
        10,
        {
            "mean_return": 100.0,
            "std_return": 5.0,
            "min_return": 90.0,
            "max_return": 110.0,
            "n_episodes": 3.0,
            "mean_episode_duration": 120.0,
        },
    )
    logger.flush()
    logger.close()

    train_rows = _read_csv(log_dir / "train.csv")
    eval_rows = _read_csv(log_dir / "eval.csv")

    assert train_rows[0] == [
        "step",
        "actor_loss",
        "critic_loss",
        "alpha_loss",
        "alpha",
        "mean_q",
        "entropy",
    ]
    assert eval_rows[0] == [
        "step",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "n_episodes",
        "mean_episode_duration",
    ]

    assert train_rows[1][0] == "10"
    assert eval_rows[1][0] == "10"


def test_logger_appends_without_duplicate_headers(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    logger = RunLogger(log_dir)
    logger.log_train(
        1,
        {
            "actor_loss": 0.1,
            "critic_loss": 0.2,
            "alpha_loss": 0.3,
            "alpha": 0.4,
            "mean_q": 0.5,
            "entropy": 0.6,
        },
    )
    logger.close()

    logger = RunLogger(log_dir)
    logger.log_train(
        2,
        {
            "actor_loss": 0.2,
            "critic_loss": 0.3,
            "alpha_loss": 0.4,
            "alpha": 0.5,
            "mean_q": 0.6,
            "entropy": 0.7,
        },
    )
    logger.close()

    rows = _read_csv(log_dir / "train.csv")
    assert rows[0][0] == "step"
    assert len(rows) == 3


def test_logger_events_jsonl(tmp_path: Path) -> None:
    utils._CACHED_TIMESTAMP = "20240101_010203"
    log_dir = tmp_path / "run"
    logger = RunLogger(log_dir)

    logger.log_event("startup", {"seed": 123, "env": "InvertedPendulum-v5"})
    logger.close()

    lines = (log_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])

    assert payload["timestamp"] == "20240101_010203"
    assert payload["event"] == "startup"
    assert payload["seed"] == 123
    assert payload["env"] == "InvertedPendulum-v5"
