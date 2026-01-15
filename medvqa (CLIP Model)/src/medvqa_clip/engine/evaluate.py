import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, Optional

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0

def _update_counts(tp, pred_cnt, true_cnt, y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):

    y_true = y_true.detach()
    y_pred = y_pred.detach()

    match = (y_true == y_pred)
    tp_add   = torch.bincount(y_true[match], minlength=num_classes).cpu()
    pred_add = torch.bincount(y_pred,        minlength=num_classes).cpu()
    true_add = torch.bincount(y_true,        minlength=num_classes).cpu()

    tp += tp_add
    pred_cnt += pred_add
    true_cnt += true_add

def _f1_from_counts(tp, pred_cnt, true_cnt, mode: str, pos_idx: Optional[int] = None) -> float:

    tp = tp.float()
    fp = pred_cnt.float() - tp
    fn = true_cnt.float() - tp

    denom = (2 * tp + fp + fn)
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))

    if mode == "binary_yes" and pos_idx is not None:
        return float(f1[pos_idx].item())

    active = (pred_cnt + true_cnt) > 0
    if active.any():
        return float(f1[active].mean().item())
    return 0.0

@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    vocabs: Optional[dict] = None,
    show_progress: bool = True,
    loss_fn: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:

    model.eval()

    total_n = 0
    total_correct = 0
    total_loss_sum = 0.0

    # per-task stats + counts
    per_task = {}   # task -> dict
    counts = {}     # task -> {tp,pred_cnt,true_cnt}

    it = tqdm(loader, desc="Evaluating", leave=True) if show_progress else loader

    for batch in it:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)

        tasks = batch["tasks"]                 # list[str]
        y = batch["labels"].to(device)         # [B]

        feats = model.forward_features(pixel_values, input_ids, attn)

        # group indices by task
        task_to_idx = defaultdict(list)
        for i, t in enumerate(tasks):
            task_to_idx[str(t)].append(i)

        pred = torch.empty_like(y)

        # per-task forward
        for t, idxs in task_to_idx.items():
            idx = torch.tensor(idxs, device=device)
            logits = model.forward_task(t, feats.index_select(0, idx))
            pred_t = logits.argmax(dim=-1)
            pred.index_copy_(0, idx, pred_t)

            C = logits.size(-1)
            if t not in per_task:
                per_task[t] = {"n": 0, "correct": 0, "loss_sum": 0.0, "num_classes": C}
                counts[t] = {
                    "tp": torch.zeros(C, dtype=torch.long),
                    "pred_cnt": torch.zeros(C, dtype=torch.long),
                    "true_cnt": torch.zeros(C, dtype=torch.long),
                }

            y_t = y.index_select(0, idx)

            _update_counts(
                counts[t]["tp"], counts[t]["pred_cnt"], counts[t]["true_cnt"],
                y_t, pred_t, num_classes=C
            )

            if loss_fn is not None:
                n_sub = idx.numel()
                per_task[t]["loss_sum"] += float(loss_fn(logits, y_t).item() * n_sub)

        ok = (pred == y)
        bs = y.size(0)

        total_n += int(bs)
        total_correct += int(ok.sum().item())

        # per-task EM accounting
        for i, t in enumerate(tasks):
            t = str(t)
            per_task[t]["n"] += 1
            per_task[t]["correct"] += int(ok[i].item())

        # overall loss (sum over samples)
        if loss_fn is not None:
            batch_loss_sum = 0.0
            for t, idxs in task_to_idx.items():
                idx = torch.tensor(idxs, device=device)
                logits = model.forward_task(t, feats.index_select(0, idx))
                y_t = y.index_select(0, idx)
                batch_loss_sum += loss_fn(logits, y_t).item() * idx.numel()
            total_loss_sum += batch_loss_sum

        if show_progress:
            it.set_postfix(em=f"{total_correct/max(1,total_n):.4f}")

    # finalize metrics
    per_task_out = {}
    weighted_f1_sum = 0.0

    for t, v in per_task.items():
        n = v["n"]
        em = _safe_div(v["correct"], n)
        loss = _safe_div(v["loss_sum"], n) if loss_fn is not None else None

        # always compute macro-f1
        f1_macro = _f1_from_counts(counts[t]["tp"], counts[t]["pred_cnt"], counts[t]["true_cnt"], mode="macro")
        f1 = f1_macro
        f1_mode = "macro"

        # if binary yes/no, compute F1_yes when possible
        C = v["num_classes"]
        if vocabs is not None and t in vocabs and C == len(vocabs[t]["itos"]) and C == 2:
            itos = [str(x).lower() for x in vocabs[t]["itos"]]
            if "yes" in itos:
                pos_idx = itos.index("yes")
                f1 = _f1_from_counts(counts[t]["tp"], counts[t]["pred_cnt"], counts[t]["true_cnt"],
                                     mode="binary_yes", pos_idx=pos_idx)
                f1_mode = "f1_yes"

        per_task_out[t] = {
            "em": em,
            "f1": f1,
            "f1_macro": f1_macro,
            "f1_mode": f1_mode,
            "loss": loss,
            "n": n
        }

        weighted_f1_sum += f1 * n

    overall = {
        "em": _safe_div(total_correct, total_n),
        "f1": _safe_div(weighted_f1_sum, total_n),
        "loss": _safe_div(total_loss_sum, total_n) if loss_fn is not None else None,
        "n": total_n
    }

    return {"overall": overall, "per_task": per_task_out}
