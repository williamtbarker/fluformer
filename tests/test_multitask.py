import pytest
import torch

from fluformer import MultiTaskConfig, MultiTaskMLP
from fluformer.models.multitask import multitask_loss, task_accuracy
from fluformer.training import multitask_train_step


def tiny_config() -> MultiTaskConfig:
    return MultiTaskConfig(
        input_dim=12,
        task_sizes={"host": 3, "subtype": 4},
        hidden_dims=(16, 8),
        dropout=0.0,
    )


def test_multitask_shapes_loss_and_step():
    torch.manual_seed(1)

    model = MultiTaskMLP(tiny_config())
    x = torch.randn(6, 12)
    targets = {
        "host": torch.randint(0, 3, (6,)),
        "subtype": torch.randint(0, 4, (6,)),
    }

    logits = model(x)

    assert logits["host"].shape == (6, 3)
    assert logits["subtype"].shape == (6, 4)

    loss, pieces = multitask_loss(logits, targets)

    assert torch.isfinite(loss)
    assert set(pieces) == {"host", "subtype"}
    assert set(task_accuracy(logits, targets)) == {
        "host",
        "subtype",
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = multitask_train_step(
        model,
        optimizer,
        x,
        targets,
    )

    assert metrics["loss"] > 0
    assert "loss/host" in metrics
    assert "loss/subtype" in metrics
    assert "accuracy/host" in metrics
    assert "accuracy/subtype" in metrics


@pytest.mark.parametrize(
    "features",
    [
        torch.randn(12),
        torch.randn(2, 11),
        torch.randn(2, 12, 1),
    ],
)
def test_multitask_rejects_invalid_feature_shapes(features):
    model = MultiTaskMLP(tiny_config())

    with pytest.raises(ValueError, match="expected features"):
        model(features)


def test_multitask_loss_requires_identical_task_names():
    logits = {
        "host": torch.randn(4, 3),
        "subtype": torch.randn(4, 4),
    }
    targets = {
        "host": torch.zeros(4, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="identical task names"):
        multitask_loss(logits, targets)


def test_multitask_loss_rejects_empty_task_mapping():
    with pytest.raises(ValueError, match="at least one task"):
        multitask_loss({}, {})


def test_multitask_loss_supports_weights_and_label_smoothing():
    torch.manual_seed(8)

    logits = {
        "host": torch.randn(5, 3),
        "subtype": torch.randn(5, 4),
    }
    targets = {
        "host": torch.randint(0, 3, (5,)),
        "subtype": torch.randint(0, 4, (5,)),
    }

    total, pieces = multitask_loss(
        logits,
        targets,
        label_smoothing=0.1,
        task_weights={"host": 2.0, "subtype": 0.5},
    )

    expected = 2.0 * pieces["host"] + 0.5 * pieces["subtype"]

    assert torch.allclose(total.detach(), expected)
    assert torch.isfinite(total)


def test_unspecified_task_weight_defaults_to_one():
    torch.manual_seed(10)

    logits = {
        "host": torch.randn(4, 3),
        "subtype": torch.randn(4, 4),
    }
    targets = {
        "host": torch.randint(0, 3, (4,)),
        "subtype": torch.randint(0, 4, (4,)),
    }

    total, pieces = multitask_loss(
        logits,
        targets,
        task_weights={"host": 2.0},
    )

    expected = 2.0 * pieces["host"] + pieces["subtype"]

    assert torch.allclose(total.detach(), expected)


def test_task_accuracy_matches_known_predictions():
    logits = {
        "task": torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [3.0, 1.0],
                [0.0, 5.0],
            ]
        )
    }
    targets = {
        "task": torch.tensor([0, 1, 1, 1])
    }

    accuracy = task_accuracy(logits, targets)

    assert accuracy["task"] == pytest.approx(0.75)


def test_state_dict_roundtrip_preserves_outputs():
    torch.manual_seed(12)

    original = MultiTaskMLP(tiny_config()).eval()
    restored = MultiTaskMLP(tiny_config()).eval()
    restored.load_state_dict(original.state_dict())

    features = torch.randn(3, 12)

    with torch.no_grad():
        first = original(features)
        second = restored(features)

    for task in first:
        assert torch.allclose(first[task], second[task])
