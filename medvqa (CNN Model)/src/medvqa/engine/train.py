import os
from collections import Counter
import torch
from tqdm import tqdm

from medvqa.engine.evaluate import evaluate_two_head_emf1
from medvqa.utils.io import write_json

def compute_open_class_weights(train_loader, num_classes: int, device):
    cnt = Counter()
    ds = train_loader.dataset
    for i in range(len(ds)):
        ex = ds[i]
        if ex["is_closed_q"] == 1:
            continue
        cnt[ex["open_label"]] += 1

    w = torch.ones(num_classes, dtype=torch.float)
    for c in range(num_classes):
        w[c] = 1.0 / max(1, cnt.get(c, 0))
    w = w / w.mean()
    return w.to(device)

def compute_closed_class_weights(train_loader, device):
    cnt = Counter()
    ds = train_loader.dataset
    for i in range(len(ds)):
        ex = ds[i]
        if ex["is_closed_q"] != 1:
            continue
        y = ex["closed_label"]
        if y == -1:
            continue
        cnt[int(y)] += 1

    w = torch.ones(2, dtype=torch.float)
    for c in range(2):
        w[c] = 1.0 / max(1, cnt.get(c, 0))
    w = w / w.mean()
    return w.to(device)

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int,
    exp_dir: str,
    logger,
    config: dict | None = None,
    alpha_closed: float = 1.0,
    ignore_open_other: bool = False,
    other_open_id: int | None = None,
):
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    if config is not None:
        write_json(config, os.path.join(exp_dir, "config.json"))

    history = []
    history_path = os.path.join(exp_dir, "history.json")

    # Weighted CE (same as your current training)
    num_open_classes = model.open_head.out_features
    w_open = compute_open_class_weights(train_loader, num_open_classes, device)
    w_closed = compute_closed_class_weights(train_loader, device)

    ce_open = torch.nn.CrossEntropyLoss(weight=w_open)
    ce_closed = torch.nn.CrossEntropyLoss(weight=w_closed)

    best = -1.0
    best_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    last_path = os.path.join(exp_dir, "checkpoints", "last.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=True)

        running_loss = 0.0
        steps = 0

        for batch in pbar:
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            open_y = batch["open_label"].to(device)
            closed_y = batch["closed_label"].to(device)
            is_closed_q = batch["is_closed_q"].to(device).bool()

            optimizer.zero_grad(set_to_none=True)

            logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

            loss = None

            # OPEN part
            mask_o = ~is_closed_q
            if mask_o.any():
                lo = logits_open[mask_o]
                yo = open_y[mask_o]
                loss_open = ce_open(lo, yo)
                loss = loss_open if loss is None else (loss + loss_open)

            # CLOSED part (valid yes/no only)
            mask_c = is_closed_q & (closed_y != -1)
            if mask_c.any():
                lc = logits_closed[mask_c]
                yc = closed_y[mask_c]
                loss_c = ce_closed(lc, yc)
                loss = (alpha_closed * loss_c) if loss is None else (loss + alpha_closed * loss_c)

            if loss is None:
                continue

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=running_loss / max(1, steps))

        train_loss = running_loss / max(1, steps)

        # Evaluate EM/F1 on train+val (optional: you can skip train eval if slow)
        train_metrics = evaluate_two_head_emf1(
            model, train_loader, device,
            show_progress=False,
            ignore_open_other=ignore_open_other,
            other_open_id=other_open_id,
        )
        val_metrics = evaluate_two_head_emf1(
            model, val_loader, device,
            show_progress=False,
            ignore_open_other=ignore_open_other,
            other_open_id=other_open_id,
        )

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(entry)
        write_json(history, history_path)

        logger(
            exp_dir,
            f"Epoch {epoch}: loss={train_loss:.4f} | "
            f"train_overall_em={train_metrics['overall']['em']:.4f} train_overall_f1={train_metrics['overall']['f1']:.4f} | "
            f"val_overall_em={val_metrics['overall']['em']:.4f} val_overall_f1={val_metrics['overall']['f1']:.4f} | "
            f"val_open_em={val_metrics['open']['em']:.4f} val_open_f1m={val_metrics['open']['f1_macro']:.4f} | "
            f"val_closed_em={val_metrics['closed']['em']:.4f} val_closed_f1yes={val_metrics['closed']['f1_yes']:.4f}"
        )

        # Save BEST by overall F1 (recommended)
        best_metric = val_metrics["overall"]["f1"]
        if best_metric > best:
            best = best_metric
            torch.save(model.state_dict(), best_path)
            logger(exp_dir, f"Saved BEST model to {best_path} (best_val_overall_f1={best:.4f})")

        torch.save(model.state_dict(), last_path)

    return best_path
