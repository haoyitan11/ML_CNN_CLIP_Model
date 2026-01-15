import os
import torch
from tqdm import tqdm
from collections import defaultdict

from medvqa_clip.engine.evaluate import evaluate
from medvqa_clip.utils.io import write_json

def train(model, train_loader, val_loader, optimizer, device, epochs, exp_dir, logger, config=None, vocabs=None):
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    if config is not None:
        write_json(config, os.path.join(exp_dir, "config.json"))

    history = []
    history_path = os.path.join(exp_dir, "history.json")

    best = -1.0
    best_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    last_path = os.path.join(exp_dir, "checkpoints", "last.pt")

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=True)

        loss_sum = 0.0
        correct_sum = 0
        n_sum = 0

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)

            tasks = batch["tasks"]
            y = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            feats = model.forward_features(pixel_values, input_ids, attn)

            task_to_idx = defaultdict(list)
            for i, t in enumerate(tasks):
                task_to_idx[str(t)].append(i)

            pred = torch.empty_like(y)
            bs = y.size(0)
            batch_loss_sum = 0.0

            for t, idxs in task_to_idx.items():
                idx = torch.tensor(idxs, device=device)
                logits = model.forward_task(t, feats.index_select(0, idx))
                y_t = y.index_select(0, idx)

                pred.index_copy_(0, idx, logits.argmax(dim=-1))
                batch_loss_sum = batch_loss_sum + (loss_fn(logits, y_t) * idx.numel())

            loss = batch_loss_sum / max(1, bs)
            loss.backward()
            optimizer.step()

            ok = (pred == y)
            loss_sum += float(loss.item() * bs)
            correct_sum += int(ok.sum().item())
            n_sum += int(bs)

            pbar.set_postfix(loss=loss_sum/max(1,n_sum), em=correct_sum/max(1,n_sum))

        train_loss = loss_sum / max(1, n_sum)

        # Compute train/val EM+F1 using evaluator
        train_metrics = evaluate(model, train_loader, device, vocabs=vocabs, show_progress=False, loss_fn=loss_fn)
        val_metrics   = evaluate(model, val_loader, device, vocabs=vocabs, show_progress=False, loss_fn=loss_fn)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_em": train_metrics["overall"]["em"],
            "train_f1": train_metrics["overall"]["f1"],
            "val_loss": val_metrics["overall"]["loss"],
            "val_em": val_metrics["overall"]["em"],
            "val_f1": val_metrics["overall"]["f1"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(entry)
        write_json(history, history_path)

        logger(
            exp_dir,
            f"Epoch {epoch}: "
            f"train_loss={entry['train_loss']:.4f} train_em={entry['train_em']:.4f} train_f1={entry['train_f1']:.4f} | "
            f"val_loss={entry['val_loss']:.4f} val_em={entry['val_em']:.4f} val_f1={entry['val_f1']:.4f}"
        )

        # Choose best checkpoint by F1 (recommended) OR EM
        if entry["val_f1"] > best:
            best = entry["val_f1"]
            torch.save(model.state_dict(), best_path)
            logger(exp_dir, f"Saved BEST to {best_path} (best_val_f1={best:.4f})")

        torch.save(model.state_dict(), last_path)

    return best_path
