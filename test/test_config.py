from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sac import config, utils


def _base_config_dict() -> dict:
    return {
        "env": "InvertedPendulum-v5",
        "seed": 0,
        "total_steps": 1000,
        "gamma": 0.99,
        "tau": 0.005,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "lr_alpha": 3e-4,
        "alpha_init": 0.2,
        "target_entropy": None,
        "batch_size": 256,
        "replay_buffer_size": 10000,
        "gradient_clip_norm": 1.0,
        "hidden_dim": 256,
        "hidden_layers": 2,
        "activation": "relu",
        "warmup_steps": 100,
        "train_freq": 1,
        "updates_per_step": 1,
        "eval_freq": 500,
        "eval_episodes": 5,
        "checkpoint_freq": 500,
    }


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    data = _base_config_dict()
    path = tmp_path / "config.yaml"
    _write_yaml(path, data)
    loaded = config.load_yaml(path)

    assert loaded["env"] == "InvertedPendulum-v5"
    assert loaded["hidden_layers"] == 2


def test_build_config_overrides_and_runtime(tmp_path: Path) -> None:
    utils._CACHED_TIMESTAMP = "20240101_010203"
    data = _base_config_dict()
    path = _write_yaml(tmp_path / "config.yaml", data)

    cfg = config.build_config(path, {"env": "Hopper-v5", "seed": 42})

    assert cfg.env == "Hopper-v5"
    assert cfg.seed == 42
    assert cfg.timestamp == "20240101_010203"
    assert cfg.log_dir.as_posix() == "logs/Hopper_v5/run_20240101_010203"
    assert cfg.checkpoint_dir.as_posix() == "checkpoints/Hopper_v5/run_20240101_010203"


def test_save_config_round_trip(tmp_path: Path) -> None:
    utils._CACHED_TIMESTAMP = "20240101_010203"
    data = _base_config_dict()
    path = _write_yaml(tmp_path / "config.yaml", data)

    cfg = config.build_config(path, {})
    out_path = tmp_path / "saved.yaml"
    config.save_config(cfg, out_path)

    saved = config.load_yaml(out_path)
    assert saved["env"] == "InvertedPendulum-v5"
    assert saved["timestamp"] == "20240101_010203"
    assert saved["log_dir"] == "logs/InvertedPendulum_v5/run_20240101_010203"


def test_build_config_validation_errors(tmp_path: Path) -> None:
    data = _base_config_dict()
    data["gamma"] = 1.5
    path = _write_yaml(tmp_path / "config.yaml", data)

    with pytest.raises(ValueError, match="gamma"):
        config.build_config(path, {})


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
