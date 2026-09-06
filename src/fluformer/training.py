from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .models.multitask import multitask_loss, task_accuracy


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve CPU/CUDA/MPS without hard-coding a workstation-specific backend."""

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def multitask_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    targets: Mapping[str, Tensor],
    *,
    label_smoothing: float = 0.0,
    max_grad_norm: float | None = 5.0,
) -> dict[str, float]:
    """Perform one optimization step for a compatible multitask model."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss, per_task = multitask_loss(logits, targets, label_smoothing=label_smoothing)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    metrics = {"loss": float(loss.detach().item())}
    metrics.update({f"loss/{name}": float(value.item()) for name, value in per_task.items()})
    metrics.update({f"accuracy/{name}": value for name, value in task_accuracy(logits, targets).items()})
    return metrics
