from datasets import load_dataset
from medvqa_clip.data.dedup import clean_dataset_splits

#load HuggingFace dataset
def load_hf_dataset(dataset_name: str, clean: bool = True):
    #download / load dataset from HuggingFace
    ds = load_dataset(dataset_name)
    
    #if caller doesnt want cleaning, return dataset
    if not clean:
        return ds
    
    #clean splits
    train, test, val = clean_dataset_splits(ds, do_dedup=True, remove_overlap=True)
    
    #rebuild output 
    out = {}
    if train is not None:
        out["train"] = train
    if test is not None:
        out["test"] = test
    if val is not None:
        out["validation"] = val
    return out
