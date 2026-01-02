import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import io, json, hashlib
from tqdm import tqdm
from src.medvqa_clip.data.load_dataset import load_hf_dataset

def image_md5(pil_img) -> str:
    #Compute MD5 hash
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

def export_images(dataset_name: str, out_dir: str, save_meta: bool = False):
    #output folder that will contain image files
    out_images = os.path.join(out_dir, "images")
    os.makedirs(out_images, exist_ok=True)

    #load dataset
    ds = load_hf_dataset(dataset_name, clean=True)
    
    #seen maps image-hash
    seen = {}
    
    #meta will store per-row metadata
    meta = []

    #loop over all splits
    for split_name in ds.keys():
        split = ds[split_name]
        
        #progress bar load
        for i, ex in enumerate(tqdm(split, desc=f"Export {split_name}")):
            #read image from dataset example
            img = ex["image"]
            #convert to RGB for consistent hashing + saving
            if hasattr(img, "convert"):
                img = img.convert("RGB")
                
            #compute hash for deduplication
            h = image_md5(img)
            
            #save only if this exact image hash never seen before
            if h not in seen:
                filename = f"{h}.png"
                img.save(os.path.join(out_images, filename))
                seen[h] = filename
                
            #optionally store metadata linking this dataset row to image file
            if save_meta:
                meta.append({
                    "split": split_name,
                    "row_id": i,
                    "image_file": seen[h],
                    "question": str(ex.get("question","")),
                    "answer": str(ex.get("answer","")),
                })

    #display message
    print(f"Done. Unique images saved: {len(seen)}")
    
    #save metadata JSON
    if save_meta:
        with open(os.path.join(out_images, "annotations_export.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--out_dir", type=str, default="images")
    ap.add_argument("--save_meta", action="store_true")
    args = ap.parse_args()
    export_images(args.dataset_name, args.out_dir, save_meta=args.save_meta)
