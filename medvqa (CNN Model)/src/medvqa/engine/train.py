import os
from collections import Counter
import torch
from tqdm import tqdm

from medvqa.engine.evaluate import evaluate_two_head
from medvqa.utils.io import write_json

def compute_open_class_weights(train_loader, num_classes: int, device):
    #Compute inverse-frequency weights for OPEN-answer classes
    cnt = Counter()
    ds = train_loader.dataset
    for i in range(len(ds)):
        ex = ds[i]
        #Skip closed (yes/no) questions when counting open-answer labels
        if ex["is_closed_q"] == 1:
            continue
        cnt[ex["open_label"]] += 1
        
    #Build weight vector: W[C] = 1 / count(c)
    w = torch.ones(num_classes, dtype=torch.float)
    for c in range(num_classes):
        w[c] = 1.0 / max(1, cnt.get(c, 0))
        
    #Normalize weights to average weight is 1.0
    w = w / w.mean()
    return w.to(device)

def compute_closed_class_weights(train_loader, device):
    #Compute inverse-frequency weights for closed (yes/no) labels
    cnt = Counter()
    ds = train_loader.dataset
    for i in range(len(ds)):
        ex = ds[i]
        #Only count closed questions
        if ex["is_closed_q"] != 1:
            continue
        y = ex["closed_label"]
        #Skip samples without a valid yes/no label
        if y == -1:
            continue
        cnt[int(y)] += 1
        
    #Two classes (eg. 0 = no, 1 = yes)
    w = torch.ones(2, dtype=torch.float)
    for c in range(2):
        w[c] = 1.0 / max(1, cnt.get(c, 0))
        
    #Normalize weights for stable loss scaling
    w = w / w.mean()
    return w.to(device)

def train(model, train_loader, val_loader, optimizer, device, epochs: int, exp_dir: str, logger, config: dict | None = None, alpha_closed: float = 1.0):
    #Create checkpoint folder
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    if config is not None:
        write_json(config, os.path.join(exp_dir, "config.json"))

    #Training history saved for each epoch
    history = []
    history_path = os.path.join(exp_dir, "history.json")

    #Prepare class weights for open-head and closed-head losses
    num_open_classes = model.open_head.out_features
    w_open = compute_open_class_weights(train_loader, num_open_classes, device)
    w_closed = compute_closed_class_weights(train_loader, device)

    #Weighted cross-entropy losses for imbalance handling
    ce_open = torch.nn.CrossEntropyLoss(weight=w_open)
    ce_closed = torch.nn.CrossEntropyLoss(weight=w_closed)

    #Track best model checkpoint (based on validation open top-1 accuracy)
    best = -1.0
    best_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    last_path = os.path.join(exp_dir, "checkpoints", "last.pt")

    for epoch in range(1, epochs + 1):
        #Set model to training mode
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=True)

        #Running stats for this epoch
        running_loss = 0.0
        steps = 0

        #Accuracy tracking (open/closed separately)
        open_correct = open_total = 0
        closed_correct = closed_total = 0

        for batch in pbar:
            #Move batch to device
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            #labels + closed/open mask
            open_y = batch["open_label"].to(device)
            closed_y = batch["closed_label"].to(device)
            is_closed_q = batch["is_closed_q"].to(device).bool()

            #reset gradients
            optimizer.zero_grad(set_to_none=True)
            
            #Forward pass (two heads)
            logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

            #Loss will combine open + closed parts
            loss = None

            #Open question loss/accuracy
            mask_o = ~is_closed_q
            if mask_o.any():
                lo = logits_open[mask_o]
                yo = open_y[mask_o]
                loss_open = ce_open(lo, yo)
                loss = loss_open if loss is None else (loss + loss_open)
                pred_o = lo.argmax(dim=-1)
                open_correct += (pred_o == yo).sum().item()
                open_total += yo.size(0)

            #Closed question loss/accuracy
            #Only include valid closed labels
            mask_c = is_closed_q & (closed_y != -1)
            if mask_c.any():
                lc = logits_closed[mask_c]
                yc = closed_y[mask_c]
                loss_c = ce_closed(lc, yc)
                
                #Optionally weight the closed loss contribution
                loss = (alpha_closed * loss_c) if loss is None else (loss + alpha_closed * loss_c)
                
                pred_c = lc.argmax(dim=-1)
                closed_correct += (pred_c == yc).sum().item()
                closed_total += yc.size(0)

            #if batch has no usable samples, skip
            if loss is None:
                continue

            #Backprop + optimizer step
            loss.backward()
            optimizer.step()

            #Update running stats
            running_loss += float(loss.item())
            steps += 1

            #Update progress display
            pbar.set_postfix(
                loss=running_loss / max(1, steps),
                open_acc=open_correct / max(1, open_total),
                closed_acc=closed_correct / max(1, closed_total),
            )

        #End of epoch: compute validation metrics
        train_loss = running_loss / max(1, steps)
        val_metrics = evaluate_two_head(model, val_loader, device, topk=5, show_progress=False)

        #Store and save training history
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_open_acc": open_correct / max(1, open_total),
            "train_closed_acc": closed_correct / max(1, closed_total),
            "val": val_metrics,
        }
        history.append(entry)
        write_json(history, history_path)

        #Choose best model by validation open top-1 accuracy
        best_metric = val_metrics["open"]["acc_top1"]

        logger(
            exp_dir,
            f"Epoch {epoch}: loss={train_loss:.4f} | "
            f"train_open_acc={entry['train_open_acc']:.4f} train_closed_acc={entry['train_closed_acc']:.4f} | "
            f"val_open_acc1={val_metrics['open']['acc_top1']:.4f} "
            f"val_open_acc5={val_metrics['open']['acc_top5']:.4f} | "
            f"val_closed_acc={val_metrics['closed']['acc']:.4f}"
        )

        #Save best checkpoint
        if best_metric > best:
            best = best_metric
            torch.save(model.state_dict(), best_path)
            logger(exp_dir, f"Saved BEST model to {best_path} (best_open_acc1={best:.4f})")

        torch.save(model.state_dict(), last_path)

    return best_path
