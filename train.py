import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import create_loaders
from src.models import build_model
from src.utils import get_device, load_config, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, leave=False, desc="train" if training else "validate")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", choices=["gcnn", "cnn"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use 3 epochs for a fast smoke run")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs
    if args.quick:
        config["epochs"] = 3

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_loaders(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    model = build_model(config["model"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_{config['model']}.pth"
    history_path = output_dir / f"history_{config['model']}.csv"
    best_accuracy = 0.0
    rows = []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        rows.append(row)
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train acc {train_acc:.2f}% | val acc {val_acc:.2f}%"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch, best_accuracy, config
            )
            print(f"Saved: {checkpoint_path}")

        with open(history_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"Run evaluation: python evaluate.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
