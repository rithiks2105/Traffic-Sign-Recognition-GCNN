import argparse
import time

import torch

from src.models import build_model
from src.utils import count_parameters, get_device


def benchmark(model, device, image_size=48, warmup=30, runs=200):
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    for name in ["cnn", "gcnn"]:
        model = build_model(name).to(device)
        latency = benchmark(model, device, args.image_size, runs=args.runs)
        params = count_parameters(model)
        try:
            from thop import profile
            sample = torch.randn(1, 3, args.image_size, args.image_size, device=device)
            macs, _ = profile(model, inputs=(sample,), verbose=False)
            macs_text = f"{macs / 1e6:.2f} M"
        except Exception:
            macs_text = "install thop"
        print(
            f"{name.upper():5s} | parameters: {params / 1e6:.3f} M | "
            f"MACs: {macs_text} | latency: {latency:.3f} ms/image"
        )


if __name__ == "__main__":
    main()
