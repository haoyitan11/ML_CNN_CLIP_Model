from _setup_path import *  # noqa: F401,F403
import argparse
from medvqa.utils.seed import set_seed
from medvqa.data.build_vocab import build_and_save_vocab

def main():
    #Entry point for the script
    
    #CLI argument
    ap = argparse.ArgumentParser()
    
    #HuggingFace dataset identifier
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    
    #Keep only top-K most frequent normalized answers in the training set
    ap.add_argument("--top_k", type=int, default=300)
    
    #Output path
    ap.add_argument("--out_path", type=str, default="outputs/answer_vocab.json")
    
    #Random seed used
    ap.add_argument("--seed", type=int, default=42)
    
    #Parse the command-line arguments 
    args = ap.parse_args()

    #Set seeds
    set_seed(args.seed)
    build_and_save_vocab(args.dataset_name, args.top_k, args.out_path, args.seed)
    print(f"Saved vocab to {args.out_path} (top_k={args.top_k})")

if __name__ == "__main__":
    main()
