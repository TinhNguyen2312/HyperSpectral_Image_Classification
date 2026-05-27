import torch

from logger import MetricLogger, SmoothedValue
from loss import compute_loss
from metrics import compute_metrics


def _unpack_batch(batch):
    if isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("images"))
        targets = batch.get("targets", batch.get("labels"))
        return inputs, targets
    inputs, targets = batch
    return inputs, targets


def _forward_model(model, inputs):
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, dict):
        outputs = outputs.get("out", outputs.get("output", outputs.get("logits")))
    return outputs


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    print_freq=10,
    log_file="",
    eval_train=True,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    criterion = getattr(args, "criterion", None)
    if criterion is None:
        raise ValueError("args.criterion must be set before training.")

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        inputs, targets = _unpack_batch(batch)
        inputs = inputs.to(args.device)
        targets = targets.to(args.device).long()

        outputs = _forward_model(model, inputs)
        total_loss, loss_dict = compute_loss(criterion, outputs, targets)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            metric_logger.update(lr=param_group["lr"])

        for k, v in loss_dict.items():
            metric_logger.update(**{f"{k}_loss": v})

        if eval_train:
            metrics = compute_metrics(outputs, targets, getattr(args, "num_classes", None))
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (float, int)):
                    metric_logger.update(**{metric_name: metric_value})

    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    data_loader,
    model,
    epoch,
    print_freq=100,
    log_file="",
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    header = f"Test: [{epoch}]"
    criterion = getattr(args, "criterion", None)

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            inputs, targets = _unpack_batch(batch)
            inputs = inputs.to(args.device)
            targets = targets.to(args.device).long()

            outputs = _forward_model(model, inputs)
            if criterion is not None:
                _, loss_dict = compute_loss(criterion, outputs, targets)
                for k, v in loss_dict.items():
                    metric_logger.update(**{f"{k}_loss": v})

            metrics = compute_metrics(outputs, targets, getattr(args, "num_classes", None))
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (float, int)):
                    metric_logger.update(**{metric_name: metric_value})

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
