import numpy as np
import torch
import torch.nn as nn


def test_compute_metrics_returns_hsi_classification_scores():
    from metrics import compute_metrics

    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 2])

    results = compute_metrics(logits, targets, n_classes=3)

    assert np.isclose(results["accuracy"], 75.0)
    assert np.isclose(results["AA"], (1.0 + 1.0 + 0.5) / 3.0 * 100.0)
    assert results["Confusion matrix"].shape == (3, 3)
    assert "Kappa" in results


def test_cross_entropy_loss_returns_total_and_loss_dict():
    from loss import build_criterion, compute_loss

    criterion = build_criterion({"name": "cross_entropy"})
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 1])

    total_loss, loss_dict = compute_loss(criterion, logits, targets)

    assert total_loss.requires_grad
    assert set(loss_dict.keys()) == {"ce", "total"}
    assert torch.isclose(loss_dict["total"], total_loss.detach())


def test_optimizer_and_scheduler_build_from_config():
    from optimizer import build_optimizer, build_scheduler

    model = nn.Linear(4, 2)
    config = {
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.01},
        "scheduler": {"name": "steplr", "step_size": 2, "gamma": 0.5},
    }

    optimizer = build_optimizer(config, model)
    scheduler, scheduler_name = build_scheduler(config, optimizer)

    assert optimizer.param_groups[0]["lr"] == 0.001
    assert scheduler_name == "steplr"
    assert scheduler.step_size == 2


def test_train_one_epoch_accepts_dct_style_batches():
    from opt import train_one_epoch
    from loss import build_criterion

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    class Args:
        device = torch.device("cpu")
        criterion = build_criterion({"name": "cross_entropy"})

    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data_loader = [
        {
            "inputs": torch.randn(3, 4),
            "targets": torch.tensor([0, 1, 0]),
        }
    ]

    stats = train_one_epoch(
        Args(),
        model,
        data_loader,
        optimizer,
        epoch=0,
        print_freq=1,
    )

    assert "total_loss" in stats
    assert "accuracy" in stats
