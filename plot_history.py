from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

output = Path("outputs")
graphs = output / "graphs"
graphs.mkdir(parents=True, exist_ok=True)

gcnn = pd.read_csv(output / "history_gcnn.csv")
cnn = pd.read_csv(output / "history_cnn.csv")

plt.figure(figsize=(12, 7))
plt.plot(gcnn["epoch"], gcnn["train_accuracy"], marker="o", label="GCNN Train")
plt.plot(gcnn["epoch"], gcnn["val_accuracy"], marker="o", label="GCNN Validation")
plt.plot(cnn["epoch"], cnn["train_accuracy"], marker="s", label="CNN Train")
plt.plot(cnn["epoch"], cnn["val_accuracy"], marker="s", label="CNN Validation")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(graphs / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 7))
plt.plot(gcnn["epoch"], gcnn["train_loss"], marker="o", label="GCNN Train Loss")
plt.plot(gcnn["epoch"], gcnn["val_loss"], marker="o", label="GCNN Validation Loss")
plt.plot(cnn["epoch"], cnn["train_loss"], marker="s", label="CNN Train Loss")
plt.plot(cnn["epoch"], cnn["val_loss"], marker="s", label="CNN Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(graphs / "loss_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved graphs in {graphs}")
