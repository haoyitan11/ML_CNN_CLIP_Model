import hashlib
import io
from typing import Tuple
from medvqa.utils.text import normalize_answer

def image_md5(pil_img) -> str:
    #convert image to bytes(PNG) and return as MD5 hash string
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

def example_key(ex) -> Tuple[str, str, str]:
    #Build a unique key for one dataset example : (image_hash, question, answer)
    img = ex["image"]
    if hasattr(img, "convert"):
        #ensure consistent hashing
        img = img.convert("RGB")
    h = image_md5(img)
    
    #normalize question
    q = str(ex.get("question", "")).strip().lower()
    
    #normalize answer
    a = normalize_answer(ex.get("answer", ""))
    return (h, q, a)

def dedup_split(hf_split):
    #Remove duplicated (image,question,answer) examples within a split
    seen = set()
    keep = []
    for i in range(len(hf_split)):
        k = example_key(hf_split[i])
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    if len(keep) == len(hf_split):
        #nothing removed
        return hf_split
    #return filtered dataset
    return hf_split.select(keep)

def remove_train_test_overlap(train_split, test_split):
    #Remove training examples that also appear in the test split
    test_keys = set()
    for i in range(len(test_split)):
        test_keys.add(example_key(test_split[i]))
    keep = []
    for i in range(len(train_split)):
        if example_key(train_split[i]) not in test_keys:
            keep.append(i)
    if len(keep) == len(train_split):
        #nothing removed
        return train_split
    return train_split.select(keep)

def clean_splits(ds, do_dedup: bool = True, remove_overlap: bool = True):
    #Extract train/test/validation splits if they exist
    train = ds["train"] if "train" in ds else None
    test = ds["test"] if "test" in ds else None
    val = ds["validation"] if "validation" in ds else None

    #Deduplicate inside each split
    if do_dedup and train is not None:
        train = dedup_split(train)
    if do_dedup and test is not None:
        test = dedup_split(test)
    if do_dedup and val is not None:
        val = dedup_split(val)

    #remove overlap between train and test
    if remove_overlap and train is not None and test is not None:
        train = remove_train_test_overlap(train, test)

    #return cleaned splits
    return train, test, val
