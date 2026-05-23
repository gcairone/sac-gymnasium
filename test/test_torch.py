import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim


def test_torch_basic_ops():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = a + b
    assert torch.allclose(c, torch.tensor([5.0, 7.0, 9.0]))

    assert c.device.type == "cpu"

    x = torch.tensor(2.0, requires_grad=True)
    y = x * x + 3.0
    y.backward()
    assert pytest.approx(x.grad.item()) == 4.0


def test_nn_forward_backward():
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    X = torch.randn(8, 3)
    y = torch.randn(8, 1)

    params_before = [p.clone().detach() for p in model.parameters()]

    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)

    optimizer.step()

    assert any((p_before - p).abs().sum().item() > 0 for p_before, p in zip(params_before, model.parameters()))

    for _ in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss).all()


def test_torch_save_load_state_dict():
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_dict, "/tmp/test_state_dict.pt")

    loaded_state_dict = torch.load("/tmp/test_state_dict.pt")
    model.load_state_dict(loaded_state_dict["model"])
    optimizer.load_state_dict(loaded_state_dict["optimizer"])

    # Model parameters should match exactly
    for p1, p2 in zip(model.parameters(), loaded_state_dict["model"].values()):
        assert p1.equal(p2)

    # Optimizer state dicts can contain nested dicts (not only tensors).
    # Check that the top-level structure and param_groups match instead
    opt_sd = optimizer.state_dict()
    loaded_opt_sd = loaded_state_dict["optimizer"]
    assert set(opt_sd.keys()) == set(loaded_opt_sd.keys())
    assert len(opt_sd["param_groups"]) == len(loaded_opt_sd["param_groups"])

    for g1, g2 in zip(opt_sd["param_groups"], loaded_opt_sd["param_groups"]):
        for key in ("lr", "momentum", "weight_decay", "dampening", "nesterov"):
            if key in g1 or key in g2:
                assert g1.get(key) == g2.get(key)

    # Ensure the state mapping has the same number of entries
    assert len(opt_sd["state"]) == len(loaded_opt_sd["state"]) 