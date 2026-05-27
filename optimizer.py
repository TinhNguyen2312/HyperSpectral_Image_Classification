import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR, ReduceLROnPlateau, StepLR


def build_optimizer(config, model):
    """Build optimizer based on DCT-IML-style configuration."""
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config["name"].lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=optimizer_config.get("weight_decay", 0),
            nesterov=optimizer_config.get("nesterov", False),
        )
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config.get("momentum", 0.0),
            weight_decay=optimizer_config.get("weight_decay", 0),
        )
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def build_scheduler(config, optimizer, last_epoch=-1):
    """Build scheduler based on DCT-IML-style configuration."""
    scheduler_config = config.get("scheduler", {"name": "none"})
    scheduler_name = scheduler_config.get("name", "none").lower()

    if scheduler_name in {"none", "null"}:
        return None, scheduler_name

    if scheduler_name == "cosineannealinglr":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["T_max"],
            eta_min=scheduler_config.get("eta_min", 0),
            last_epoch=last_epoch,
        )
    elif scheduler_name == "steplr":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config.get("gamma", 0.1),
            last_epoch=last_epoch,
        )
    elif scheduler_name == "multisteplr":
        scheduler = MultiStepLR(
            optimizer,
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config.get("gamma", 0.1),
            last_epoch=last_epoch,
        )
    elif scheduler_name == "reducelronplateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.1),
            patience=scheduler_config.get("patience", 10),
        )
    elif scheduler_name == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch - config.get("epoch_count", 0) - config.get("niter", 100)
            ) / float(config.get("niter_decay", 100) + 1)
            return lr_l

        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler, scheduler_name
