import torch
from tqdm import tqdm

def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0

def _update_multiclass_counts(tp, pred_cnt, true_cnt, y_true, y_pred, num_classes: int):

    y_true = y_true.detach()
    y_pred = y_pred.detach()

    match = (y_true == y_pred)
    tp_add   = torch.bincount(y_true[match], minlength=num_classes).cpu()
    pred_add = torch.bincount(y_pred,        minlength=num_classes).cpu()
    true_add = torch.bincount(y_true,        minlength=num_classes).cpu()

    tp += tp_add
    pred_cnt += pred_add
    true_cnt += true_add

def _macro_f1_from_counts(tp, pred_cnt, true_cnt, ignore_index: int | None = None) -> float:
    tp = tp.float()
    fp = pred_cnt.float() - tp
    fn = true_cnt.float() - tp

    denom = (2 * tp + fp + fn)
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.zeros_like(denom))

    active = (pred_cnt + true_cnt) > 0
    if ignore_index is not None and 0 <= ignore_index < active.numel():
        active[ignore_index] = False

    if active.any():
        return float(f1[active].mean().item())
    return 0.0

def _binary_f1_yes_from_counts(tp, pred_cnt, true_cnt, pos_idx: int = 1) -> float:
    # F1 for positive class (default: 1 = "yes")
    tp = float(tp[pos_idx].item())
    fp = float((pred_cnt[pos_idx] - tp).item())
    fn = float((true_cnt[pos_idx] - tp).item())
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0

@torch.no_grad()
def evaluate_two_head_emf1(
    model,
    loader,
    device,
    show_progress: bool = True,
    ignore_open_other: bool = False,
    other_open_id: int | None = None,
):

    model.eval()

    # OPEN stats
    open_total = 0
    open_correct = 0
    open_num_classes = model.open_head.out_features
    open_tp = torch.zeros(open_num_classes, dtype=torch.long)
    open_pred_cnt = torch.zeros(open_num_classes, dtype=torch.long)
    open_true_cnt = torch.zeros(open_num_classes, dtype=torch.long)

    # CLOSED stats 
    closed_total = 0
    closed_correct = 0
    closed_tp = torch.zeros(2, dtype=torch.long)
    closed_pred_cnt = torch.zeros(2, dtype=torch.long)
    closed_true_cnt = torch.zeros(2, dtype=torch.long)

    it = tqdm(loader, desc="Evaluating", leave=True) if show_progress else loader

    for batch in it:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        open_y = batch["open_label"].to(device)
        closed_y = batch["closed_label"].to(device)
        is_closed_q = batch["is_closed_q"].to(device).bool()

        logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

        #OPEN: only on open questions
        mask_o = ~is_closed_q
        if mask_o.any():
            lo = logits_open[mask_o]
            yo = open_y[mask_o]
            pred_o = lo.argmax(dim=-1)

            open_total += int(yo.size(0))
            open_correct += int((pred_o == yo).sum().item())

            _update_multiclass_counts(open_tp, open_pred_cnt, open_true_cnt, yo, pred_o, open_num_classes)

        #CLOSED: only on closed questions with valid labels
        mask_c = is_closed_q & (closed_y != -1)
        if mask_c.any():
            lc = logits_closed[mask_c]
            yc = closed_y[mask_c]
            pred_c = lc.argmax(dim=-1)

            closed_total += int(yc.size(0))
            closed_correct += int((pred_c == yc).sum().item())

            _update_multiclass_counts(closed_tp, closed_pred_cnt, closed_true_cnt, yc, pred_c, 2)

        if show_progress:
            it.set_postfix(
                open_em=f"{open_correct/max(1,open_total):.4f}",
                closed_em=f"{closed_correct/max(1,closed_total):.4f}",
            )

    # OPEN F1
    ignore_idx = other_open_id if (ignore_open_other and other_open_id is not None) else None
    open_f1_macro = _macro_f1_from_counts(open_tp, open_pred_cnt, open_true_cnt, ignore_index=ignore_idx)

    # CLOSED F1
    closed_f1_yes = _binary_f1_yes_from_counts(closed_tp, closed_pred_cnt, closed_true_cnt, pos_idx=1)
    closed_f1_macro = _macro_f1_from_counts(closed_tp, closed_pred_cnt, closed_true_cnt)

    # OVERALL
    total_n = open_total + closed_total
    overall_em = _safe_div(open_correct + closed_correct, total_n)
    overall_f1 = _safe_div(open_f1_macro * open_total + closed_f1_yes * closed_total, total_n)

    return {
        "overall": {"em": overall_em, "f1": overall_f1, "n": total_n},
        "open": {"em": _safe_div(open_correct, open_total), "f1_macro": open_f1_macro, "n": open_total},
        "closed": {"em": _safe_div(closed_correct, closed_total), "f1_yes": closed_f1_yes, "f1_macro": closed_f1_macro, "n": closed_total},
    }
