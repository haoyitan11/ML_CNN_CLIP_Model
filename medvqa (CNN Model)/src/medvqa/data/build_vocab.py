from collections import Counter
from tqdm import tqdm
from medvqa.data.load_dataset import load_hf_dataset
from medvqa.utils.io import write_json
from medvqa.utils.text import normalize_answer

#Build an answer vocabulary from a training dataset split
def build_answer_vocab_from_train(train_split, top_k: int = 300, show_progress: bool = True):
    
    #Dictionary-like object mapping answer_string
    cnt = Counter()
    #Progress bar
    it = tqdm(train_split, desc="Building answer vocab") if show_progress else train_split
    
    #Iterate through each training example
    for ex in it:
        #read the raw answer text and normalize it to canonicab form
        ans = normalize_answer(ex.get("answer", ""))
        if ans:
            cnt[ans] += 1
            
    #Extract the top_K answers by frequency
    most = [a for a, _ in cnt.most_common(top_k)]
    itos = most + ["OTHER"]
    stoi = {a: i for i, a in enumerate(itos)}
    return {"stoi": stoi, "itos": itos}

def build_and_save_vocab(dataset_name: str, top_k: int, out_path: str, seed: int = 42, show_progress: bool = True):
    #Load dataset, build vocab, save as JSON, return vocab dict
    
    #load dataset
    ds = load_hf_dataset(dataset_name, clean=True)
    
    #train split
    train = ds["train"] if "train" in ds else list(ds.values())[0]
    
    #Build vocabulary from training example
    vocab = build_answer_vocab_from_train(train, top_k=top_k, show_progress=show_progress)
    
    #Save vocabulary JSON
    write_json(vocab, out_path)
    return vocab
