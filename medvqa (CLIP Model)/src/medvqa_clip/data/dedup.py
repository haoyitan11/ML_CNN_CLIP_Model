import hashlib, io
from typing import Dict, Tuple
from medvqa_clip.utils.text import normalize_answer

#compute MD5 hash of an image
def _img_md5(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

#build unique key for an example
def _example_key(ex) -> Tuple[str, str, str]:
    #extract PIL image
    img = ex["image"]
    
    #convert to RGB for consistent hashing
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    
    #hash the image pixels
    h = _img_md5(img)
    
    #normalize question
    q = str(ex.get("question", "")).strip().lower()
    
    #normalize answer 
    a = normalize_answer(ex.get("answer", ""))
    
    #composite key: image + question + answer
    return (h, q, a)

#remove duplicate sample inside a single dataset split
def dedup_split(hf_split):
    seen = set()
    keep_indices = []
    for i in range(len(hf_split)):
        ex = hf_split[i]
        k = _example_key(ex)
        
        #skip if duplicate
        if k in seen:
            continue
        
        #keep if new
        seen.add(k)
        keep_indices.append(i)
        
    #if nothing removed, return original split unchanged
    if len(keep_indices) == len(hf_split):
        return hf_split

    #otherwise return filtered split
    return hf_split.select(keep_indices)

#remove overlap between train and test
def remove_train_test_overlap(train_split, test_split):
    #build set of all keys in test split
    test_keys = set()
    for i in range(len(test_split)):
        test_keys.add(_example_key(test_split[i]))
        
    #keep only train examples not present in test_keys
    keep_indices = []
    for i in range(len(train_split)):
        if _example_key(train_split[i]) not in test_keys:
            keep_indices.append(i)
            
    #if nothing removed, return original train split
    if len(keep_indices) == len(train_split):
        return train_split
    
    #otherwise return filtered training split
    return train_split.select(keep_indices)

#clean dataset splits
def clean_dataset_splits(ds: Dict, do_dedup=True, remove_overlap=True):
    #extract splits if present
    train = ds["train"] if "train" in ds else None
    test = ds["test"] if "test" in ds else None
    val = ds["validation"] if "validation" in ds else None

    #deduplicate within each split
    if train is not None and do_dedup:
        train = dedup_split(train)
    if test is not None and do_dedup:
        test = dedup_split(test)
    if val is not None and do_dedup:
        val = dedup_split(val)

    #remove train/test overlap
    if train is not None and test is not None and remove_overlap:
        train = remove_train_test_overlap(train, test)

    #return cleaned splits
    return train, test, val
