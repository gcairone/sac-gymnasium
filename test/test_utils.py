import random
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sac import utils


def test_soft_update():
    target = torch.nn.Linear(3, 2, bias=False)
    source = torch.nn.Linear(3, 2, bias=False)

    torch.nn.init.zeros_(target.weight)
    torch.nn.init.ones_(source.weight)

    utils.soft_update(target, source, tau=0.5)

    expected = torch.full_like(target.weight, 0.5)
    assert torch.allclose(target.weight, expected)


def test_init_weights_orthogonal():
    layer = torch.nn.Linear(4, 4, bias=True)
    utils.init_weights_orthogonal(layer)

    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))
    weight = layer.weight.detach()
    prod = weight @ weight.t()
    assert torch.allclose(prod, torch.eye(4), atol=1e-5)


def test_init_weights_uniform():
    layer = torch.nn.Linear(3, 2, bias=True)
    utils.init_weights_uniform(layer, scale=1e-3)

    assert layer.weight.abs().max().item() <= 1e-3 + 1e-6
    assert layer.bias.abs().max().item() <= 1e-3 + 1e-6


def test_get_timestamp_cached_format():
    utils._CACHED_TIMESTAMP = None
    ts1 = utils.get_timestamp()
    ts2 = utils.get_timestamp()

    assert ts1 == ts2
    assert re.match(r"^\d{8}_\d{6}$", ts1)


def test_paths_and_env_sanitization():
    ts = "20240101_010203"
    assert utils.sanitize_env_name("HalfCheetah-v5") == "HalfCheetah_v5"
    assert utils.make_log_dir("HalfCheetah-v5", ts).as_posix() == (
        "logs/HalfCheetah_v5/run_20240101_010203"
    )
    assert utils.make_checkpoint_dir("HalfCheetah-v5", ts).as_posix() == (
        "checkpoints/HalfCheetah_v5/run_20240101_010203"
    )


def test_set_seed_reproducibility():
    utils.set_seed(123)
    py_1 = random.random()
    np_1 = np.random.rand()
    torch_1 = torch.rand(1).item()

    utils.set_seed(123)
    py_2 = random.random()
    np_2 = np.random.rand()
    torch_2 = torch.rand(1).item()

    assert py_1 == py_2
    assert np_1 == np_2
    assert torch_1 == torch_2

