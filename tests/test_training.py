import torch

from fluformer import MultiTaskConfig, MultiTaskMLP
from fluformer.training import multitask_train_step, resolve_device


def tiny_model() -> MultiTaskMLP:
    return MultiTaskMLP(
        MultiTaskConfig(
            input_dim=6,
            task_sizes={"task": 3},
            hidden_dims=(8,),
            dropout=0.0,
        )
    )


def test_explicit_device_is_respected():
    assert resolve_device("cpu") == torch.device("cpu")


def test_cuda_is_preferred_when_available(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: False,
    )

    assert resolve_device() == torch.device("cuda")


def test_mps_is_selected_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: True,
    )

    assert resolve_device() == torch.device("mps")


def test_cpu_is_fallback_when_accelerators_are_unavailable(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        torch.backends.mps,
        "is_available",
        lambda: False,
    )

    assert resolve_device() == torch.device("cpu")


def test_train_step_can_disable_gradient_clipping(monkeypatch):
    torch.manual_seed(13)

    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    features = torch.randn(5, 6)
    targets = {
        "task": torch.randint(0, 3, (5,))
    }

    def should_not_run(*args, **kwargs):
        raise AssertionError(
            "gradient clipping should be disabled"
        )

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        should_not_run,
    )

    metrics = multitask_train_step(
        model,
        optimizer,
        features,
        targets,
        max_grad_norm=None,
    )

    assert metrics["loss"] > 0
    assert "loss/task" in metrics
    assert "accuracy/task" in metrics


def test_train_step_uses_gradient_clipping_when_configured(
    monkeypatch,
):
    torch.manual_seed(14)

    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    features = torch.randn(5, 6)
    targets = {
        "task": torch.randint(0, 3, (5,))
    }

    called = {}

    original = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        called["max_norm"] = max_norm
        return original(
            parameters,
            max_norm,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        recording_clip,
    )

    multitask_train_step(
        model,
        optimizer,
        features,
        targets,
        max_grad_norm=2.5,
    )

    assert called["max_norm"] == 2.5
