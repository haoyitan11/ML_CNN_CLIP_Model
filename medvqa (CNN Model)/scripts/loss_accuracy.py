import json
from pathlib import Path
import matplotlib.pyplot as plt

history_path = Path(
    r"D:\UM\Document\WOA7001 ADVANCED MACHINE LEARNING\Alternative assessment\Coding\Coding (CNN and CLIP Model)\medvqa (CNN Model)\outputs\exp_cnn\history.json"
)

if not history_path.exists():
    raise FileNotFoundError(f"history.json not found at: {history_path}")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

if not isinstance(history, list):
    raise ValueError("Expected history.json to be a LIST of epoch dicts.")

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def to_pct(x):
    return None if x is None else float(x) * 100.0

def filter_xy(x, y):
    xx, yy = [], []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        xx.append(xi)
        yy.append(yi)
    return xx, yy

# Extract epochs
epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
max_epoch = max(epochs) if epochs else 0
xticks = list(range(1, max_epoch + 1))  # 1..40

# Accuracy metrics (overall EM + overall F1) -> %
train_em = [to_pct(safe_get(h, "train", "overall", "em")) for h in history]
val_em   = [to_pct(safe_get(h, "val",   "overall", "em")) for h in history]

train_f1 = [to_pct(safe_get(h, "train", "overall", "f1")) for h in history]
val_f1   = [to_pct(safe_get(h, "val",   "overall", "f1")) for h in history]


# Loss metrics
train_loss = [h.get("train_loss") for h in history]
val_loss   = [h.get("val_loss") for h in history]  # likely None (not logged)

out_dir = history_path.parent
acc_png  = out_dir / "accuracy_curve.png"
loss_png = out_dir / "loss_curve.png"

# 1) ACCURACY (EM + F1)
x_te, y_te = filter_xy(epochs, train_em)
x_ve, y_ve = filter_xy(epochs, val_em)
x_tf, y_tf = filter_xy(epochs, train_f1)
x_vf, y_vf = filter_xy(epochs, val_f1)

plt.figure(figsize=(12, 5))
if y_te: plt.plot(x_te, y_te, marker="o", label="Baseline 1 : (CLIP Model) (Train EM)")
if y_ve: plt.plot(x_ve, y_ve, marker="o", label="Baseline 1 : (CLIP Model) (Val EM)")
if y_tf: plt.plot(x_tf, y_tf, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Train F1)")
if y_vf: plt.plot(x_vf, y_vf, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Val F1)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train & Validation Accuracy for CLIP Model")
plt.grid(True, alpha=0.3)
if max_epoch > 0:
    plt.xticks(xticks)
plt.ylim(0, 100)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(acc_png, dpi=200)
plt.close()

# 2) LOSS
x_tl, y_tl = filter_xy(epochs, train_loss)
x_vl, y_vl = filter_xy(epochs, val_loss)

plt.figure(figsize=(12, 5))
if y_tl: plt.plot(x_tl, y_tl, marker="o", label="Train Loss")
if y_vl:
    plt.plot(x_vl, y_vl, marker="o", label="Val Loss")
else:
    print("[WARN] val_loss not found in history.json (only train_loss is plotted).")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.grid(True, alpha=0.3)
if max_epoch > 0:
    plt.xticks(xticks)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(loss_png, dpi=200)
plt.close()

print("Saved:", acc_png)
print("Saved:", loss_png)
