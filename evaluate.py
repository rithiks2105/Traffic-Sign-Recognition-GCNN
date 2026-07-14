import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.data import create_loaders
from src.models import build_model
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_gcnn.pth")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", choices=["gcnn", "cnn"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_config = checkpoint.get("config", {})
    model_name = args.model or saved_config.get("model") or config.get("model", "gcnn")

    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader = create_loaders(
        data_dir=config["data_dir"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="test"):
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    accuracy = 100.0 * sum(p == t for p, t in zip(predictions, targets)) / len(targets)
    print(f"Model: {model_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(classification_report(targets, predictions, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(targets, predictions))


if __name__ == "__main__":
    main()
