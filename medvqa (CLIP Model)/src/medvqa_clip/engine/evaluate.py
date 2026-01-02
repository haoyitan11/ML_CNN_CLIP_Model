import torch
from collections import defaultdict
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, loader, device, show_progress=True):
    #Put model in evaluation mode 
    model.eval()
    
    #overall counters
    total = 0
    correct = 0
    
    #per-task counters
    per_task = defaultdict(lambda: {"n": 0, "correct": 0})

    #optional progress bar
    it = tqdm(loader, desc="Evaluating", leave=True) if show_progress else loader

    #loop over batches
    for batch in it:
        #CLIP image tensor
        pixel_values = batch["pixel_values"].to(device)
        
        #tokenized questions
        input_ids = batch["input_ids"].to(device)
        
        #attention mask may be missing depending on processor
        attn = batch.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)
            
        #task names
        tasks = batch["tasks"]
        
        #ground-truth labels
        y = batch["labels"].to(device)

        #Extract shared multimodal 
        feats = model.forward_features(pixel_values, input_ids, attn)

        # group indices by task to avoid per-sample loop
        task_to_idx = defaultdict(list)
        for i, t in enumerate(tasks):
            task_to_idx[t].append(i)

        #Prepare tensors for prediction
        pred = torch.empty_like(y)
        
        #For each task group
        for t, idxs in task_to_idx.items():
            #Convert list of indices into a tensor on the same device
            idx = torch.tensor(idxs, device=device)
            
            #Forward the features through the specific task head
            logits = model.forward_task(t, feats.index_select(0, idx))
            
            #Argmax gives predicted class ID
            pred.index_copy_(0, idx, logits.argmax(dim=-1))

        #Compute correctness mask per sample
        ok = (pred == y)
        
        #batch size
        bs = y.size(0)
        
        #update overall counters
        total += bs
        correct += ok.sum().item()

        #Update per-task counters (per sample)
        for i, t in enumerate(tasks):
            per_task[t]["n"] += 1
            per_task[t]["correct"] += int(ok[i].item())

        #update progress bar display
        if show_progress:
            it.set_postfix(acc=f"{correct/max(1,total):.4f}")

    out = {
        "overall": {"acc": correct/max(1,total), "n": total},
        "per_task": {t: {"acc": v["correct"]/max(1,v["n"]), "n": v["n"]} for t, v in per_task.items()}
    }
    return out
