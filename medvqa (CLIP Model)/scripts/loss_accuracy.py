import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
history_path = ROOT / "outputs" / "exp_clip" / "history.json"

if not history_path.exists():
    raise FileNotFoundError(f"history.json not found at: {history_path}")

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

def get_metric(h, *keys, default=None):
    for k in keys:
        if k in h and h[k] is not None:
            return h[k]
    return default

def filter_xy(x, y):
    xx, yy = [], []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        xx.append(xi)
        yy.append(yi)
    return xx, yy

def to_pct(arr):
    # keep None as None, multiply valid floats by 100
    return [None if v is None else float(v) * 100.0 for v in arr]

# ---- Parse history ----
if isinstance(history, list):
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]

    train_loss = [get_metric(h, "train_loss") for h in history]
    val_loss   = [get_metric(h, "val_loss") for h in history]

    train_em = [get_metric(h, "train_em") for h in history]
    val_em   = [get_metric(h, "val_em") for h in history]

    train_f1 = [get_metric(h, "train_f1") for h in history]
    val_f1   = [get_metric(h, "val_f1") for h in history]
else:
    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss", [])
    train_em   = history.get("train_em", [])
    val_em     = history.get("val_em", [])
    train_f1   = history.get("train_f1", [])
    val_f1     = history.get("val_f1", [])
    epochs = list(range(1, max(len(train_loss), len(train_em), len(train_f1)) + 1))

out_dir = history_path.parent
loss_png = out_dir / "loss_curve.png"
emf1_png = out_dir / "em_f1_curve.png"

max_epoch = max(epochs) if epochs else 0
xticks = list(range(1, max_epoch + 1))

# 1) LOSS PNG
x_tl, y_tl = filter_xy(epochs, train_loss)
x_vl, y_vl = filter_xy(epochs, val_loss)

plt.figure(figsize=(12, 5))
if y_tl: plt.plot(x_tl, y_tl, marker="o", label="Baseline 1 : (CNN Model) (Train)")
if y_vl: plt.plot(x_vl, y_vl, marker="o", label="Baseline 1 : (CNN Model) (Val)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss for CNN Model")
plt.grid(True, alpha=0.3)
if max_epoch > 0:
    plt.xticks(xticks)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(loss_png, dpi=200)
plt.close()


# 2) EM + F1 PNG (dual y-axis) as %
train_em_pct = to_pct(train_em)
val_em_pct   = to_pct(val_em)
train_f1_pct = to_pct(train_f1)
val_f1_pct   = to_pct(val_f1)

x_te, y_te = filter_xy(epochs, train_em_pct)
x_ve, y_ve = filter_xy(epochs, val_em_pct)
x_tf, y_tf = filter_xy(epochs, train_f1_pct)
x_vf, y_vf = filter_xy(epochs, val_f1_pct)

fig, ax1 = plt.subplots(figsize=(12, 5))

# Left axis: EM (%)
if y_te: ax1.plot(x_te, y_te, marker="o", label="Baseline 1 : (CNN Model) (Train EM)")
if y_ve: ax1.plot(x_ve, y_ve, marker="o", label="Baseline 1 : (CNN Model) (Train EM)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.grid(True, alpha=0.3)
if max_epoch > 0:
    ax1.set_xticks(xticks)
ax1.set_ylim(0, 100)  # comment this out if you don't want fixed scale

# Right axis: F1 (%)
ax2 = ax1.twinx()
if y_tf: ax2.plot(x_tf, y_tf, marker="s", linestyle="--", label="Baseline 2 : (CNN Model) (Train F1)")
if y_vf: ax2.plot(x_vf, y_vf, marker="s", linestyle="--", label="Baseline 2 : (CNN Model) (Val F1)")
ax2.set_ylim(0, 100)  # comment this out if you don't want fixed scale

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Training & Validation Accuracy for CNN Model")
plt.tight_layout()
plt.savefig(emf1_png, dpi=200)
plt.close()

print("Saved:", loss_png)
print("Saved:", emf1_png)
