import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Helpers
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

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

def parse_history(history):
    out = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_em": [],
        "val_em": [],
        "train_f1": [],
        "val_f1": [],
    }

    if isinstance(history, list):
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
        out["epochs"] = epochs

        # Loss
        out["train_loss"] = [h.get("train_loss") for h in history]
        out["val_loss"]   = [h.get("val_loss") for h in history]

        # Try CNN nested metrics first
        train_em_nested = [safe_get(h, "train", "overall", "em") for h in history]
        val_em_nested   = [safe_get(h, "val",   "overall", "em") for h in history]
        train_f1_nested = [safe_get(h, "train", "overall", "f1") for h in history]
        val_f1_nested   = [safe_get(h, "val",   "overall", "f1") for h in history]

        # If nested not present, try flat keys (CLIP-style)
        if all(v is None for v in train_em_nested) and all(v is None for v in train_f1_nested):
            out["train_em"] = [h.get("train_em") for h in history]
            out["val_em"]   = [h.get("val_em") for h in history]
            out["train_f1"] = [h.get("train_f1") for h in history]
            out["val_f1"]   = [h.get("val_f1") for h in history]
        else:
            out["train_em"] = train_em_nested
            out["val_em"]   = val_em_nested
            out["train_f1"] = train_f1_nested
            out["val_f1"]   = val_f1_nested

        # Last fallback: accuracy keys (if your CLIP history uses train_acc/val_acc)
        if all(v is None for v in out["train_em"]) and all(v is None for v in out["train_f1"]):
            train_acc = [h.get("train_accuracy") or h.get("train_acc") for h in history]
            val_acc   = [h.get("val_accuracy") or h.get("val_acc") for h in history]
            # treat "accuracy" as EM line if nothing else exists
            out["train_em"] = train_acc
            out["val_em"]   = val_acc

    elif isinstance(history, dict):
        # dict-of-lists format
        # epochs derived from longest known list
        n = max(
            len(history.get("train_loss", [])),
            len(history.get("val_loss", [])),
            len(history.get("train_em", [])),
            len(history.get("train_f1", [])),
            len(history.get("train_accuracy", []) or history.get("train_acc", [])),
            0
        )
        out["epochs"] = list(range(1, n + 1))
        out["train_loss"] = history.get("train_loss", [None]*n)
        out["val_loss"]   = history.get("val_loss", [None]*n)

        out["train_em"] = history.get("train_em", [None]*n)
        out["val_em"]   = history.get("val_em", [None]*n)
        out["train_f1"] = history.get("train_f1", [None]*n)
        out["val_f1"]   = history.get("val_f1", [None]*n)

        # fallback accuracy keys
        if all(v is None for v in out["train_em"]) and all(v is None for v in out["train_f1"]):
            out["train_em"] = history.get("train_accuracy") or history.get("train_acc") or [None]*n
            out["val_em"]   = history.get("val_accuracy") or history.get("val_acc") or [None]*n
    else:
        raise ValueError("Unsupported history format. Expected list or dict.")

    return out

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn_history", type=str, required=True, help="Path to CNN history.json")
    ap.add_argument("--clip_history", type=str, required=True, help="Path to CLIP history.json")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (default: same folder as CNN history)")
    args = ap.parse_args()

    cnn_path = Path(args.cnn_history)
    clip_path = Path(args.clip_history)

    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN history.json not found: {cnn_path}")
    if not clip_path.exists():
        raise FileNotFoundError(f"CLIP history.json not found: {clip_path}")

    cnn = parse_history(load_json(cnn_path))
    clip = parse_history(load_json(clip_path))

    out_dir = Path(args.out_dir) if args.out_dir else cnn_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_png  = out_dir / "accuracy_compare.png"
    loss_png = out_dir / "loss_compare.png"

    # Build shared xticks
    max_epoch = max(
        max(cnn["epochs"]) if cnn["epochs"] else 0,
        max(clip["epochs"]) if clip["epochs"] else 0
    )
    xticks = list(range(1, max_epoch + 1))

    # ACCURACY PNG (single chart: EM + F1 together)
    cnn_train_em = [to_pct(v) for v in cnn["train_em"]]
    cnn_val_em   = [to_pct(v) for v in cnn["val_em"]]
    cnn_train_f1 = [to_pct(v) for v in cnn["train_f1"]]
    cnn_val_f1   = [to_pct(v) for v in cnn["val_f1"]]

    clip_train_em = [to_pct(v) for v in clip["train_em"]]
    clip_val_em   = [to_pct(v) for v in clip["val_em"]]
    clip_train_f1 = [to_pct(v) for v in clip["train_f1"]]
    clip_val_f1   = [to_pct(v) for v in clip["val_f1"]]

    plt.figure(figsize=(12, 6))

    # CNN EM (solid)
    x, y = filter_xy(cnn["epochs"], cnn_train_em)
    if y: plt.plot(x, y, marker="o", linestyle="-",  label="Baseline 1 : (CNN Model) (Train EM)")
    x, y = filter_xy(cnn["epochs"], cnn_val_em)
    if y: plt.plot(x, y, marker="o", linestyle="-",  label="Baseline 1 : (CNN Model) (Val EM)")

    # CNN F1 (dashed)
    x, y = filter_xy(cnn["epochs"], cnn_train_f1)
    if y: plt.plot(x, y, marker="o", linestyle="--", label="Baseline 1 : (CNN Model) (Train F1)")
    x, y = filter_xy(cnn["epochs"], cnn_val_f1)
    if y: plt.plot(x, y, marker="o", linestyle="--", label="Baseline 1 : (CNN Model) (Val F1)")

    # CLIP EM (solid)
    x, y = filter_xy(clip["epochs"], clip_train_em)
    if y: plt.plot(x, y, marker="s", linestyle="-",  label="Baseline 2 : (CLIP Model) (Train EM)")
    x, y = filter_xy(clip["epochs"], clip_val_em)
    if y: plt.plot(x, y, marker="s", linestyle="-",  label="Baseline 2 : (CLIP Model) (Val EM)")

    # CLIP F1 (dashed)
    x, y = filter_xy(clip["epochs"], clip_train_f1)
    if y: plt.plot(x, y, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Train F1)")
    x, y = filter_xy(clip["epochs"], clip_val_f1)
    if y: plt.plot(x, y, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Val F1)")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy Comparison for CNN Model and CLIP Model")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    if max_epoch > 0:
        plt.xticks(xticks)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(acc_png, dpi=200)
    plt.close()

    # LOSS PNG (single panel)
    plt.figure(figsize=(12, 5))

    x, y = filter_xy(cnn["epochs"], cnn["train_loss"])
    if y: plt.plot(x, y, marker="o", label="Baseline 1 : (CNN Model) (Train Loss)")

    x, y = filter_xy(cnn["epochs"], cnn["val_loss"])
    if y: plt.plot(x, y, marker="o", label="Baseline 1 : (CNN Model) (Val Loss)")

    x, y = filter_xy(clip["epochs"], clip["train_loss"])
    if y: plt.plot(x, y, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Train Loss)")

    x, y = filter_xy(clip["epochs"], clip["val_loss"])
    if y: plt.plot(x, y, marker="s", linestyle="--", label="Baseline 2 : (CLIP Model) (Val Loss)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Comparison")
    plt.grid(True, alpha=0.3)
    if max_epoch > 0:
        plt.xticks(xticks)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)
    plt.close()

    print("Saved:", acc_png)
    print("Saved:", loss_png)

if __name__ == "__main__":
    main()
