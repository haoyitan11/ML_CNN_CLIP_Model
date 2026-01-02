import argparse
import io
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

#This is used to detect duplicate images
def image_md5(pil_img) -> str:
    #Create in-memory bytes buffer
    buf = io.BytesIO()
    #Save the image into the buffer
    pil_img.save(buf, format="PNG")
    #Compute and return the MD5 hash
    return hashlib.md5(buf.getvalue()).hexdigest()

def main():
    #Build command-line interface
    ap = argparse.ArgumentParser()
    
    #HuggingFace dataset name to download/load
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    
    #Output directory
    ap.add_argument("--out_dir", type=str, required=True)
    #If provided, export JSON file
    ap.add_argument("--save_meta", action="store_true")
    #Parse arguments
    args = ap.parse_args()

    #Convert output directory string to path object for easier path handling
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #Download/load dataset from HuggingFace
    ds = load_dataset(args.dataset_name)
    #Track already-saved image
    seen = {}
    #Optional metadata rows for exporting annotations
    meta = []

    #Loop over each split
    for split_name in ds.keys():
        split = ds[split_name]
        #Loop over every example in the split with progress bar
        for i, ex in enumerate(tqdm(split, desc=f"Export {split_name}")):
            #Extract PIL image from dataset example
            img = ex["image"]
            
            #Ensure RGB so hashing/saving is consistent
            if hasattr(img, "convert"):
                img = img.convert("RGB")

            #Hash the image to identify duplicates
            h = image_md5(img)
            #If image hash havent saved yet, save it once
            if h not in seen:
                filename = f"{h}.png"
                img.save(out_dir / filename)
                seen[h] = filename

            #If user requested metadata export, store the row info
            if args.save_meta:
                meta.append({
                    #Which dataset split this row come from
                    "split": split_name,
                    #Row index within that split
                    "row_id": i,
                    #Image filename
                    "image_file": seen[h],
                    #Question text
                    "question": str(ex.get("question", "")),
                    #Answer text
                    "answer": str(ex.get("answer", "")),
                })

    #Display message
    print(f"Done. Unique images saved: {len(seen)}")
    print(f"Saved to: {out_dir}")

    #If metadata was requested, write it as JSON file
    if args.save_meta:
        meta_path = out_dir / "annotations_export.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata: {meta_path}")

if __name__ == "__main__":
    
    main()
