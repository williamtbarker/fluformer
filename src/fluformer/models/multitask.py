from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from ..config import MultiTaskConfig


class MultiTaskMLP(nn.Module):
    """Shared representation with one classification head per task.

    Task names and class counts are fully configurable, allowing the same
    shared representation to support multiple related classification targets.
    """

    def __init__(self, config: MultiTaskConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        previous = config.input_dim

        for hidden in config.hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
            )
            previous = hidden

        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(previous, size)
                for name, size in config.task_sizes.items()
            }
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if (
            features.ndim != 2
            or features.shape[-1] != self.config.input_dim
        ):
            raise ValueError(
                f"expected features shaped "
                f"[batch, {self.config.input_dim}], "
                f"got {tuple(features.shape)}"
            )

        shared = self.trunk(features)
        return {
            name: head(shared)
            for name, head in self.heads.items()
        }


def multitask_loss(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    label_smoothing: float = 0.0,
    task_weights: Mapping[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return weighted summed cross-entropy and detached per-task losses."""

    if logits.keys() != targets.keys():
        raise ValueError(
            "logits and targets must contain identical task names"
        )

    if not logits:
        raise ValueError("at least one task is required")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    losses: dict[str, Tensor] = {}
    total: Tensor | None = None

    for name, prediction in logits.items():
        loss = criterion(prediction, targets[name])

        weight = (
            1.0
            if task_weights is None
            else float(task_weights.get(name, 1.0))
        )

        weighted = loss * weight
        total = (
            weighted
            if total is None
            else total + weighted
        )

        losses[name] = loss.detach()

    assert total is not None

    return total, losses


def task_accuracy(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
) -> dict[str, float]:
    """Compute simple top-1 accuracy for each task."""

    return {
        name: float(
            (
                prediction.argmax(dim=-1)
                == targets[name]
            )
            .float()
            .mean()
            .item()
        )
        for name, prediction in logits.items()
    }
