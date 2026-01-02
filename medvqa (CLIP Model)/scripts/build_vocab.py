import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
from src.medvqa_clip.utils.io import write_json

def main():
    #parse command-line arguments
    ap = argparse.ArgumentParser()
    
    #output path for vocab bundle JSON
    ap.add_argument("--out_path", type=str, default="outputs/vocabs.json")
    
    args = ap.parse_args()

    #define vocabulary
    vocabs = {
        #closed yes/no tasks
        "yesno": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "pneumonia": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "effusion": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "pneumothorax": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "cardiomegaly": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "mass": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "is_ct": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        "is_mri": {"itos": ["yes", "no"], "stoi": {"yes": 0, "no": 1}},
        
        #side classification (whether abnormal or not)
        "side": {"itos": ["left", "right", "bilateral"], "stoi": {"left": 0, "right": 1, "bilateral": 2}},
        
        #open tasks (multi-class classification heads)
        #imaging modality classification
        "modality": {"itos": ["x-ray", "ct", "mri", "ultrasound"], "stoi": {"x-ray": 0, "ct": 1, "mri": 2, "ultrasound": 3}},
        
        #image plane classification (CT/MRI)
        "plane": {"itos": ["axial", "coronal", "sagittal"], "stoi": {"axial": 0, "coronal": 1, "sagittal": 2}},
        
        #x-ray view classification
        "view": {"itos": ["pa", "ap", "lateral"], "stoi": {"pa": 0, "ap": 1, "lateral": 2}},
        
        #organ/region classification
        "organ": {"itos": ["brain", "chest", "abdomen", "kidney", "liver", "spine"], "stoi": {"brain": 0, "chest": 1, "abdomen": 2, "kidney": 3, "liver": 4, "spine": 5}},\
        
        #MRI/CT sequence-type classification
        "sequence": {"itos": ["flair", "t1", "t2", "dwi", "adc", "contrast"], "stoi": {"flair": 0, "t1": 1, "t2": 2, "dwi": 3, "adc": 4, "contrast": 5}}
    }

    #save vocab bundle to JSON
    write_json(vocabs, args.out_path)
    #display message
    print(f"Saved: {args.out_path}")

if __name__ == "__main__":
    main()
