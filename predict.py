import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data import get_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint, load_config


CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles >3.5t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware ice/snow", "Wild animals crossing",
    "End speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End no passing",
    "End no passing >3.5t",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="outputs/best_gcnn.pth")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    model = build_model(config["model"]).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    _, transform = get_transforms(config["image_size"])
    image = Image.open(Path(args.image)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = probabilities.topk(3)

    for rank, (value, index) in enumerate(zip(values, indices), start=1):
        print(f"{rank}. {CLASS_NAMES[index.item()]}: {value.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
