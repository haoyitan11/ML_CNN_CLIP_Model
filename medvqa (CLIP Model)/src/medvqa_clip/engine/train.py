import os
import torch
from tqdm import tqdm
from collections import defaultdict

from medvqa_clip.engine.evaluate import evaluate
from medvqa_clip.utils.io import write_json

#Train loop for multitaskCLIP
def train(model, train_loader, val_loader, optimizer, device, epochs, exp_dir, logger, config=None):
    #ensure checkpoint exists
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    
    #Save training configuration
    if config is not None:
        write_json(config, os.path.join(exp_dir, "config.json"))

    #Store epoch-by-epoch results
    history = []
    history_path = os.path.join(exp_dir, "history.json")

    #track best validation accuracy so far
    best = -1.0
    best_path = os.path.join(exp_dir, "checkpoints", "best.pt")
    last_path = os.path.join(exp_dir, "checkpoints", "last.pt")

    # Equal weight for all tasks by default
    ce = torch.nn.CrossEntropyLoss()

    #Epoch loop
    for epoch in range(1, epochs+1):
        #Switch model to training mode
        model.train()
        
        #Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=True)
        
        #Running sums for loss and accuracy over the epoch
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        #Batch loop
        for batch in pbar:
            #Move image tensor
            pixel_values = batch["pixel_values"].to(device)
            
            #Move tokenized question to device
            input_ids = batch["input_ids"].to(device)
            
            #Attention mask might be missing depending on processor settings
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
                
            #Tasks name for each sample in the batch
            tasks = batch["tasks"]
            
            #Ground-truth label for each sample
            y = batch["labels"].to(device)

            #Reset gradients 
            optimizer.zero_grad(set_to_none=True)

            #Extract shared fused features from CLIP
            feats = model.forward_features(pixel_values, input_ids, attn)

            # group indices per task
            task_to_idx = defaultdict(list)
            for i, t in enumerate(tasks):
                task_to_idx[t].append(i)

            #Total loss across all tasks present in the batch
            loss = 0.0
            
            #Store predictions for the whole batch
            pred = torch.empty_like(y)
            
            #for each group, compute logits + loss
            for t, idxs in task_to_idx.items():
                #convert indices to a tensor
                idx = torch.tensor(idxs, device=device)
                
                #Forward only subset of features
                logits = model.forward_task(t, feats.index_select(0, idx))
                
                #Add cross-entropy loss for this task subset
                loss = loss + ce(logits, y.index_select(0, idx))
                
                #predicted class ID
                pred.index_copy_(0, idx, logits.argmax(dim=-1))

            #Backprop + update weights
            loss.backward()
            optimizer.step()

            #update running stats
            running_loss += loss.item() * y.size(0)
            running_correct += (pred == y).sum().item()
            running_total += y.size(0)

            #update progress bar text
            pbar.set_postfix(loss=running_loss/max(1,running_total), acc=running_correct/max(1,running_total))

        #Compute epoch-level training metrics
        train_loss = running_loss/max(1,running_total)
        train_acc = running_correct/max(1,running_total)

        #Validation
        val_metrics = evaluate(model, val_loader, device, show_progress=False)
        val_acc = val_metrics["overall"]["acc"]

        #Save epoch result to history
        entry = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val": val_metrics}
        history.append(entry)
        
        #Persist history to disk every epoch
        write_json(history, history_path)

        logger(exp_dir, f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        #Save BEST checkpoint
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), best_path)
            logger(exp_dir, f"Saved BEST to {best_path} (best_val_acc={best:.4f})")

        #Save LAST checkpoint every epoch
        torch.save(model.state_dict(), last_path)

    return best_path
