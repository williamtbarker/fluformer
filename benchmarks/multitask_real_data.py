#!/usr/bin/env python3
"""Train and evaluate Fluformer's multitask classifier on precomputed embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from fluformer import MultiTaskConfig, MultiTaskMLP
from fluformer.models.multitask import multitask_loss
from fluformer.training import resolve_device

TASKS = ("host", "subtype", "clade")
LABEL_KEYS = {"host": "Y_host", "subtype": "Y_subtype", "clade": "Y_clade"}
CLASS_KEYS = {
    "host": "host_classes",
    "subtype": "subtype_classes",
    "clade": "clade_classes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-data benchmark for Fluformer MultiTaskMLP."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_multitask"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512, 256])
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate input datasets without training.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> dict:
    print(f"Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "X", "Y_host", "Y_subtype", "Y_clade", "isolate_ids",
        "host_classes", "subtype_classes", "clade_classes",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    if data["X"].ndim != 2:
        raise ValueError(f"{path}: X must be two-dimensional")
    n = data["X"].shape[0]
    for task in TASKS:
        target = data[LABEL_KEYS[task]]
        if target.shape != (n,):
            raise ValueError(
                f"{path}: {LABEL_KEYS[task]} has invalid shape {tuple(target.shape)}"
            )
    if len(data["isolate_ids"]) != n:
        raise ValueError(f"{path}: isolate_ids length does not match X")
    return data


def audit_splits(train: dict, val: dict, test: dict) -> dict:
    splits = {"train": train, "val": val, "test": test}
    feature_dims = {name: int(data["X"].shape[1]) for name, data in splits.items()}
    if len(set(feature_dims.values())) != 1:
        raise ValueError(f"feature dimensions differ across splits: {feature_dims}")
    for task in TASKS:
        class_sets = {name: tuple(data[CLASS_KEYS[task]]) for name, data in splits.items()}
        if len(set(class_sets.values())) != 1:
            raise ValueError(f"{task} class definitions differ across splits")
        expected_classes = len(train[CLASS_KEYS[task]])
        for split_name, data in splits.items():
            target = data[LABEL_KEYS[task]]
            if target.numel():
                minimum = int(target.min())
                maximum = int(target.max())
                if minimum < 0 or maximum >= expected_classes:
                    raise ValueError(
                        f"{split_name} {task} labels fall outside [0, {expected_classes - 1}]"
                    )
    id_sets = {name: set(data["isolate_ids"]) for name, data in splits.items()}
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    duplicates = {
        name: len(data["isolate_ids"]) - len(id_sets[name])
        for name, data in splits.items()
    }
    return {
        "samples": {name: int(data["X"].shape[0]) for name, data in splits.items()},
        "feature_dimensions": feature_dims,
        "classes": {task: list(train[CLASS_KEYS[task]]) for task in TASKS},
        "split_id_overlap": overlaps,
        "within_split_duplicate_ids": duplicates,
    }


def subset_split(data: dict, maximum: int | None, seed: int) -> dict:
    if maximum is None or maximum >= data["X"].shape[0]:
        return data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(data["X"].shape[0], generator=generator)[:maximum]
    selected = {"X": data["X"][indices]}
    for task in TASKS:
        selected[LABEL_KEYS[task]] = data[LABEL_KEYS[task]][indices]
    selected["isolate_ids"] = [data["isolate_ids"][index] for index in indices.tolist()]
    for task in TASKS:
        selected[CLASS_KEYS[task]] = data[CLASS_KEYS[task]]
    return selected


def make_loader(data: dict, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(data["X"], data["Y_host"], data["Y_subtype"], data["Y_clade"])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def batch_targets(batch: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {"host": batch[1], "subtype": batch[2], "clade": batch[3]}


def move_targets(targets: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in targets.items()}


@torch.no_grad()
def evaluate(
    model: MultiTaskMLP,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    true_values = {task: [] for task in TASKS}
    predictions = {task: [] for task in TASKS}
    for batch in loader:
        features = batch[0].to(device)
        targets = move_targets(batch_targets(batch), device)
        logits = model(features)
        loss, _ = multitask_loss(logits, targets, label_smoothing=label_smoothing)
        batch_n = features.shape[0]
        loss_sum += float(loss.item()) * batch_n
        sample_count += batch_n
        for task in TASKS:
            true_values[task].append(targets[task].cpu())
            predictions[task].append(logits[task].argmax(dim=-1).cpu())
    y_true = {task: torch.cat(true_values[task]).numpy() for task in TASKS}
    y_pred = {task: torch.cat(predictions[task]).numpy() for task in TASKS}
    metrics = {"loss": loss_sum / sample_count}
    for task in TASKS:
        metrics[task] = {
            "accuracy": accuracy_score(y_true[task], y_pred[task]),
            "balanced_accuracy": balanced_accuracy_score(y_true[task], y_pred[task]),
            "macro_f1": f1_score(y_true[task], y_pred[task], average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true[task], y_pred[task], average="weighted", zero_division=0),
        }
    return metrics, y_true, y_pred


def train_epoch(
    model: MultiTaskMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        features = batch[0].to(device)
        targets = move_targets(batch_targets(batch), device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss, _ = multitask_loss(logits, targets, label_smoothing=label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch_n = features.shape[0]
        loss_sum += float(loss.detach().item()) * batch_n
        sample_count += batch_n
    return loss_sum / sample_count


def mean_macro_f1(metrics: dict) -> float:
    return float(np.mean([metrics[task]["macro_f1"] for task in TASKS]))


def save_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_loss(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fluformer multitask training")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_validation_metrics(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for task in TASKS:
        ax.plot(epochs, [row[f"val_{task}_macro_f1"] for row in history], label=task.capitalize())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.set_title("Validation macro F1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test_metrics(metrics: dict, path: Path) -> None:
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    x = np.arange(len(TASKS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, metric_name in enumerate(metric_names):
        values = [metrics[task][metric_name] for task in TASKS]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=metric_name.replace("_", " ").title(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([task.capitalize() for task in TASKS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-out multitask performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_outputs(
    task: str,
    classes: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    labels = np.arange(len(classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig_size = max(6, min(12, len(classes) * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{task.capitalize()} normalized confusion matrix")
    fig.colorbar(image, ax=ax, label="Fraction")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{task}.png", dpi=180)
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_path = out_dir / f"classification_report_{task}.csv"
    with report_path.open("w", newline="") as handle:
        fieldnames = ["class", "precision", "recall", "f1-score", "support"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, values in report.items():
            if not isinstance(values, dict):
                continue
            writer.writerow(
                {
                    "class": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1-score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )


def save_predictions(
    isolate_ids: list[str],
    y_true: dict[str, np.ndarray],
    y_pred: dict[str, np.ndarray],
    classes: dict[str, list[str]],
    path: Path,
) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = ["isolate_id"]
        for task in TASKS:
            fieldnames.extend([f"{task}_true", f"{task}_pred"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, isolate_id in enumerate(isolate_ids):
            row = {"isolate_id": isolate_id}
            for task in TASKS:
                row[f"{task}_true"] = classes[task][int(y_true[task][index])]
                row[f"{task}_pred"] = classes[task][int(y_pred[task][index])]
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.train)
    val = load_split(args.val)
    test = load_split(args.test)
    audit = audit_splits(train, val, test)

    with (args.out_dir / "data_audit.json").open("w") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))

    overlap_total = sum(audit["split_id_overlap"].values())
    if overlap_total:
        raise RuntimeError(
            "Isolate IDs overlap across train/validation/test splits. Refusing to benchmark."
        )

    if args.audit_only:
        print("Audit complete; no training requested.")
        return 0

    train = subset_split(train, args.max_train, args.seed)
    val = subset_split(val, args.max_val, args.seed + 1)
    test = subset_split(test, args.max_test, args.seed + 2)

    input_dim = int(train["X"].shape[1])
    classes = {task: list(train[CLASS_KEYS[task]]) for task in TASKS}
    task_sizes = {task: len(classes[task]) for task in TASKS}

    config = MultiTaskConfig(
        input_dim=input_dim,
        task_sizes=task_sizes,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Input dimension: {input_dim}")
    print(f"Task sizes: {task_sizes}")

    model = MultiTaskMLP(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(train, args.batch_size, True, args.seed)
    val_loader = make_loader(val, args.batch_size, False, args.seed)
    test_loader = make_loader(test, args.batch_size, False, args.seed)

    history = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint = args.out_dir / "best_model.pt"
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.label_smoothing,
        )
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
            args.label_smoothing,
        )
        score = mean_macro_f1(val_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_host_accuracy": val_metrics["host"]["accuracy"],
            "val_host_macro_f1": val_metrics["host"]["macro_f1"],
            "val_subtype_accuracy": val_metrics["subtype"]["accuracy"],
            "val_subtype_macro_f1": val_metrics["subtype"]["macro_f1"],
            "val_clade_accuracy": val_metrics["clade"]["accuracy"],
            "val_clade_macro_f1": val_metrics["clade"]["macro_f1"],
            "mean_val_macro_f1": score,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"host F1 {val_metrics['host']['macro_f1']:.4f} | "
            f"subtype F1 {val_metrics['subtype']['macro_f1']:.4f} | "
            f"clade F1 {val_metrics['clade']['macro_f1']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": config.input_dim,
                        "task_sizes": config.task_sizes,
                        "hidden_dims": config.hidden_dims,
                        "dropout": config.dropout,
                    },
                    "epoch": epoch,
                    "mean_val_macro_f1": score,
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after epoch {epoch}; best epoch was {best_epoch}."
                )
                break

    training_seconds = time.perf_counter() - start_time
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        args.label_smoothing,
    )

    summary = {
        "benchmark": "real_multitask",
        "seed": args.seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_mean_validation_macro_f1": best_score,
        "training_seconds": training_seconds,
        "samples": {
            "train": len(train["isolate_ids"]),
            "validation": len(val["isolate_ids"]),
            "test": len(test["isolate_ids"]),
        },
        "test": test_metrics,
    }

    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    save_history(history, args.out_dir / "history.csv")
    plot_training_loss(history, args.out_dir / "training_loss.png")
    plot_validation_metrics(history, args.out_dir / "validation_macro_f1.png")
    plot_test_metrics(test_metrics, args.out_dir / "test_metrics.png")

    for task in TASKS:
        save_confusion_outputs(
            task,
            classes[task],
            y_true[task],
            y_pred[task],
            args.out_dir,
        )

    save_predictions(
        test["isolate_ids"],
        y_true,
        y_pred,
        classes,
        args.out_dir / "predictions.csv",
    )

    print("\nHeld-out test results")
    for task in TASKS:
        task_metrics = test_metrics[task]
        print(
            f"{task:8s} "
            f"accuracy={task_metrics['accuracy']:.4f} "
            f"balanced_accuracy={task_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={task_metrics['macro_f1']:.4f} "
            f"weighted_f1={task_metrics['weighted_f1']:.4f}"
        )

    print(f"\nResults written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
