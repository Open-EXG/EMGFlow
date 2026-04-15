import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from EMG_fidelity.models import ResNet18_1D
from EMG_fidelity.robustmodel import Crossformer1D, EMGHandNet1D, SimpleConvNet
from emgflow.config.path import get_dataset_root
from emgflow.datasets.NinaproDB2 import NinaproDB2Simple
from emgflow.datasets.NinaproDB4 import NinaproDB4Simple
from emgflow.datasets.NinaproDB7 import NinaproDB7Simple
from emgflow.datasets.utils.split import SplitSpec


def _build_model(model_name: str, num_classes: int, device: str):
    if model_name == "resnet18":
        model = ResNet18_1D(input_channels=12, num_classes=num_classes)
    elif model_name == "simpleconvnet":
        model = SimpleConvNet(num_classes=num_classes)
    elif model_name == "emghandnet":
        model = EMGHandNet1D(num_classes=num_classes)
    elif model_name == "crossformer":
        model = Crossformer1D(num_classes=num_classes, in_ch=12, in_len=400)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Train classifier for FID/IS computation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["NinaproDB2", "NinaproDB7", "NinaproDB4"],
        help="Dataset to train on",
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject ID")
    parser.add_argument("--epochs", type=int, default=75, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="Weight decay for AdamW")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "simpleconvnet", "emghandnet", "crossformer"], help="Model architecture")
    parser.add_argument("--save_name", type=str, default=None, help="Optional checkpoint file name (default: {dataset}_{model}.pth)")
    parser.add_argument("--root", type=str, default=None, help="Optional dataset root override.")
    args = parser.parse_args()

    print(f"Training Classifier on {args.dataset}")

    # 1. Setup Dataset
    if args.dataset == "NinaproDB2":
        root = args.root or get_dataset_root("NinaproDB2")
        ds_cls = NinaproDB2Simple
        num_classes = 49
    elif args.dataset == "NinaproDB7":
        root = args.root or get_dataset_root("NinaproDB7")
        ds_cls = NinaproDB7Simple
        num_classes = 40
    else:
        root = args.root or get_dataset_root("NinaproDB4")
        ds_cls = NinaproDB4Simple
        num_classes = 52

    if not root:
        raise ValueError(
            f"Dataset root for {args.dataset} is not configured. "
            "Pass --root or set the corresponding NINAPRO_*_ROOT environment variable."
        )

    # Compute stats on Train Split
    split_spec = SplitSpec(
        group_by="repetition",
        train_trials=[1, 3, 4, 6],
        val_trials=None,
        test_trials=[2, 5],
    )

    print(f"Loading Dataset from {root}...")
    ds = ds_cls(
        root=root,
        subjects=[args.subject],
        window_size=400,
        stride=100,
        denoise=False,
        wavelet_level=3,
        norm_method="zscore",
        norm_mode="per_channel",
        split_spec=split_spec,
    )

    splits = ds.split(split_spec)
    train_set = ds.subset(splits["train"])
    test_set = ds.subset(splits["test"])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Setup Model
    model = _build_model(args.model, num_classes=num_classes, device=args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 5
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    save_dir = Path(__file__).parent / "checkpoints" / args.dataset / f"subj{args.subject}"
    save_dir.mkdir(exist_ok=True, parents=True)
    save_name = args.save_name or f"{args.dataset}_{args.model}.pth"
    save_path = save_dir / save_name
    shared_log_path = Path(__file__).parent / "checkpoints" / "final_test_acc.log"
    best_test_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device).long()
            y = y - 1  # Adjust labels to start from 0
            optimizer.zero_grad()
            _, logits = model(x) 
            
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        scheduler.step()

        # Evaluate every 10 epochs and save best checkpoint.
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(args.device), y.to(args.device).long()
                    y = y - 1  # Adjust labels to start from 0
                    _, logits = model(x)
                    _, predicted = logits.max(1)
                    test_total += y.size(0)
                    test_correct += predicted.eq(y).sum().item()

            test_acc_epoch = 100 * test_correct / test_total
            print(f"[Epoch {epoch+1}] Test Acc: {test_acc_epoch:.2f}%")

            if test_acc_epoch > best_test_acc:
                best_test_acc = test_acc_epoch
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best! Saved to {save_path}")

    # Reload best checkpoint and evaluate.
    model.load_state_dict(torch.load(save_path, map_location=args.device))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device).long()
            y = y - 1  # Adjust labels to start from 0
            _, logits = model(x)
            _, predicted = logits.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()

    test_acc = 100 * test_correct / test_total
    print(f"Best Test Acc: {test_acc:.2f}%")

    shared_log_path.parent.mkdir(exist_ok=True, parents=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(shared_log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{now}\tdataset={args.dataset}\tmodel={args.model}\tsubj={args.subject}\tfinal_test_acc={test_acc:.2f}%\n"
        )
    print(f"Appended final result to shared log: {shared_log_path}")

if __name__ == "__main__":
    main()
