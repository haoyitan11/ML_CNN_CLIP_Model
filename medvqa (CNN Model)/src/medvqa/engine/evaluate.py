import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate_two_head(model, loader, device, topk: int = 5, show_progress: bool = True):
    #switch model to evaluation mode
    model.eval()

    #counters for open-ended questions
    total_o = 0
    correct1_o = 0
    correctk_o = 0

    #counters for closed-ended questions
    total_c = 0
    correct_c = 0

    #progress bar
    it = tqdm(loader, desc="Evaluating", leave=True) if show_progress else loader

    for batch in it:
        #Move inputs to correct device
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        #Ground-truth labels + mask indicating which questions are closed-ended
        open_y = batch["open_label"].to(device)
        closed_y = batch["closed_label"].to(device)
        is_closed_q = batch["is_closed_q"].to(device).bool()

        #Forward pass: model returns logits for open-head and closed-head
        logits_open, logits_closed = model(image=image, input_ids=input_ids, attention_mask=attention_mask)

        #open questions evaluation (top-1 + top-K accuracy) 
        mask_o = ~is_closed_q
        if mask_o.any():
            #open-head logits for open questions
            lo = logits_open[mask_o]           
            #open-head ground truth label 
            yo = open_y[mask_o]
            
            #top-1 prediction
            pred1 = lo.argmax(dim=-1)
            
            #avoid asking for topK > num_classes
            k = min(topk, lo.size(-1))
            
            #top-K predicted class ID
            top = lo.topk(k=k, dim=-1).indices
            
            #correct top-1
            ok1 = (pred1 == yo)
            #correct if GT appears in top-K
            okk = (top == yo.unsqueeze(1)).any(dim=1)
            total_o += yo.size(0)
            correct1_o += ok1.sum().item()
            correctk_o += okk.sum().item()

        #Closed questions evaluation (yes/no accuracy)
        #Only evaluate closed questions that actually have a valid yes/no label
        mask_c = is_closed_q & (closed_y != -1)
        if mask_c.any():
            #Closed-head logits for closed questions
            lc = logits_closed[mask_c]
            #Closed-head ground truth labels
            yc = closed_y[mask_c]
            
            #Predicted yes/no
            pred = lc.argmax(dim=-1)
            total_c += yc.size(0)
            correct_c += (pred == yc).sum().item()

        #progress bar
        if show_progress:
            it.set_postfix(open_acc1=f"{correct1_o/max(1,total_o):.4f}", closed_acc=f"{correct_c/max(1,total_c):.4f}")

    #Return final metrics as dict
    return {
        "open": {"acc_top1": correct1_o/max(1,total_o), "acc_top5": correctk_o/max(1,total_o), "n": total_o},
        "closed": {"acc": correct_c/max(1,total_c), "n": total_c},
    }
