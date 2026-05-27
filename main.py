import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from logger import Logger
from loss import build_criterion
from models.get_model import get_model
from opt import evaluate_fn, train_one_epoch
from optimizer import build_optimizer, build_scheduler
from utils.dataset import HSIDataset, load_mat_hsi, sample_gt
from utils.utils import (
    check_state_dict,
    cleanup_distributed,
    count_model_parameters,
    get_rank,
    save_on_master,
    setup_distributed,
    split_info_print,
)


def get_args_parser():
    parser = argparse.ArgumentParser("HSI classification", add_help=False)
    parser.add_argument("--cfg_path", default="configs/hsi.yaml", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--patch_size", default=None, type=int)
    parser.add_argument("--batch-size", "--bs", dest="batch_size", default=None, type=int)
    parser.add_argument("--epochs", "--epoch", dest="epochs", default=None, type=int)
    parser.add_argument("--ratio", default=None, type=float)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser


def apply_cli_overrides(args, cfg):
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.dataset_name is not None:
        cfg["data"]["dataset_name"] = args.dataset_name
    if args.dataset_dir is not None:
        cfg["data"]["dataset_dir"] = args.dataset_dir
    if args.patch_size is not None:
        cfg["data"]["patch_size"] = args.patch_size
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.ratio is not None:
        cfg["data"]["train_val_ratio"] = args.ratio
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.output_dir is not None:
        cfg["training"]["model_dir"] = args.output_dir
    return cfg


def build_dataloaders(cfg, args, is_distributed):
    cfg_data = cfg["data"]
    image, gt, labels = load_mat_hsi(cfg_data["dataset_name"], cfg_data["dataset_dir"])
    trainval_gt, test_gt = sample_gt(
        gt,
        float(cfg_data["train_val_ratio"]),
        int(cfg["training"].get("seed", 202401)),
    )
    train_gt, val_gt = sample_gt(
        trainval_gt,
        0.5,
        int(cfg["training"].get("seed", 202401)),
    )

    if get_rank() == 0:
        split_info_print(train_gt, val_gt, test_gt, labels)

    patch_size = int(cfg_data["patch_size"])
    train_set = HSIDataset(image, train_gt, patch_size=patch_size, data_aug=True, return_dict=True)
    val_set = HSIDataset(image, val_gt, patch_size=patch_size, data_aug=False, return_dict=True)
    test_set = HSIDataset(image, test_gt, patch_size=patch_size, data_aug=False, return_dict=True)

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_distributed else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if is_distributed else None

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg_data.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": HSIDataset.data_collator,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, sampler=val_sampler, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, sampler=test_sampler, shuffle=False, **loader_kwargs)

    return image, gt, labels, train_loader, val_loader, test_loader, train_sampler


def main(args, cfg):
    cfg = apply_cli_overrides(args, cfg)
    is_distributed, rank, local_rank, _ = setup_distributed()

    model_dir = Path(cfg["training"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = str(model_dir / f"run_{timestamp}.log")
    if rank == 0:
        Logger(log_file)

    device = torch.device(
        f"cuda:{local_rank}"
        if is_distributed and torch.cuda.is_available()
        else (
            args.device
            if args.device == "cpu" or str(args.device).startswith("cuda")
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    )

    seed = int(cfg["training"].get("seed", 202401)) + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    image, gt, labels, train_loader, val_loader, test_loader, train_sampler = build_dataloaders(
        cfg, args, is_distributed
    )
    num_classes = len(labels)

    model_name = cfg["model"]["name"]
    patch_size = int(cfg["data"]["patch_size"])
    model = get_model(model_name, cfg["data"]["dataset_name"], patch_size).to(device)

    if is_distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    model_for_params = model.module if is_distributed else model
    n_parameters = count_model_parameters(model_for_params)
    if rank == 0:
        print(f"Model: {model_name}")
        print(f"Dataset: {cfg['data']['dataset_name']}")
        print(f"Patch size: {patch_size}")
        print(f"Number of parameters: {n_parameters:,}")

    criterion = build_criterion(cfg["training"].get("loss", {"name": "cross_entropy"})).to(device)
    args.criterion = criterion
    args.device = device
    args.num_classes = num_classes
    args.output_dir = str(model_dir)

    optimizer = build_optimizer(cfg["training"]["optimization"], model_for_params)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    scheduler_last_epoch = -1
    if args.resume:
        if rank == 0:
            print(f"Resume from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if check_state_dict(model_for_params, checkpoint["model_state_dict"]):
            ret = model_for_params.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            raise ValueError("Model and state dict are different")

        args.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        scheduler_last_epoch = int(checkpoint.get("epoch", -1))
        if rank == 0:
            print("Missing keys: \n", "\n".join(ret.missing_keys))
            print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    scheduler_config = cfg["training"]["optimization"].setdefault("scheduler", {"name": "none"})
    if scheduler_config.get("name", "none").lower() == "cosineannealinglr":
        scheduler_config["T_max"] = int(cfg["training"]["epochs"])
    scheduler, scheduler_type = build_scheduler(
        cfg["training"]["optimization"], optimizer, last_epoch=scheduler_last_epoch
    )

    if args.resume and not args.eval:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if args.eval:
        test_results = evaluate_fn(
            args,
            test_loader,
            model,
            epoch=0,
            print_freq=args.print_freq,
            log_file=log_file,
        )
        if rank == 0:
            print(f"Test accuracy: {test_results.get('accuracy', 0.0):.3f}")
        return

    if rank == 0:
        print(f"Training on {device}")
        print(f"Start training for {cfg['training']['epochs']} epochs")

    start_time = time.time()
    best_acc = -1.0

    for epoch in range(args.start_epoch, int(cfg["training"]["epochs"])):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_results = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_file=log_file,
        )

        if scheduler is not None and scheduler_type != "reducelronplateau":
            scheduler.step()

        val_results = evaluate_fn(
            args,
            val_loader,
            model,
            epoch,
            print_freq=args.print_freq,
            log_file=log_file,
        )

        if scheduler is not None and scheduler_type == "reducelronplateau":
            scheduler.step(val_results.get("total_loss", 0.0))

        current_acc = val_results.get("accuracy", 0.0)
        is_best = current_acc > best_acc
        best_acc = max(best_acc, current_acc)

        checkpoint = {
            "model_state_dict": model_for_params.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
            "epoch": epoch,
            "cfg": cfg,
        }
        save_on_master(checkpoint, model_dir / f"checkpoint_{epoch}.pth")
        if is_best:
            save_on_master(checkpoint, model_dir / "best_checkpoint.pth")

        log_results = {
            **{f"train_{k}": v for k, v in train_results.items()},
            **{f"val_{k}": v for k, v in val_results.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        if rank == 0:
            print(f"* VAL accuracy {current_acc:.3f} Best accuracy {best_acc:.3f}")
            with (model_dir / "log.txt").open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_results) + "\n")

    test_results = evaluate_fn(
        args,
        test_loader,
        model,
        epoch=int(cfg["training"]["epochs"]),
        print_freq=args.print_freq,
        log_file=log_file,
    )
    if rank == 0:
        print(f"Final test accuracy: {test_results.get('accuracy', 0.0):.3f}")
        total_time = time.time() - start_time
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))

    cleanup_distributed()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser("HSI classification", parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.cfg_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)
