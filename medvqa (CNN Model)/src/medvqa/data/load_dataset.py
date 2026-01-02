from datasets import load_dataset
from medvqa.data.dedup import clean_splits

def load_hf_dataset(dataset_name: str, clean: bool = True):
    #Load HuggingFace dataset
    ds = load_dataset(dataset_name)
    
    #If cleaning is disabled, return raw dataset
    if not clean:
        return ds
    
    #Clean the dataset splits: deduplicate + remove train/test overlap
    train, test, val = clean_splits(ds, do_dedup=True, remove_overlap=True)
    
    #Rebuild output dict with only the splits
    out = {}
    if train is not None:
        out["train"] = train
    if test is not None:
        out["test"] = test
    if val is not None:
        out["validation"] = val
        
    #return cleaned splits in a simple dictionary
    return out
