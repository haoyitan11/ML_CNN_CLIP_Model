from medvqa_clip.utils.text import normalize_answer

#yes/no normalization
YES = {"yes", "y", "true", "present"}
NO  = {"no", "n", "false", "absent"}

#convert a raw answer into yes/no label
def to_yesno(ans: str):
    a = normalize_answer(ans)
    #Direct matches
    if a in YES:
        return "yes"
    if a in NO:
        return "no"
    
    #Common variants like "no abnormality", "yes pneumonia"
    if a.startswith("no "):
        return "no"
    if a.startswith("yes "):
        return "yes"
    
    #Not a recognizable yes/no answer
    return None

#convert a raw answer into one of a few modality labels
def to_modality(ans: str):
    a = normalize_answer(ans)
    
    #x-ray mentions
    if "x-ray" in a or "xray" in a:
        return "x-ray"
    
    #CT mentions
    if "ct" in a:
        return "ct"
    
    #MRI mentions
    if "mri" in a or "mr" in a:
        return "mri"
    
    #Ultrasound mentions
    if "ultrasound" in a or "us" == a:
        return "ultrasound"
    return None

#convert a raw answer into imaging plane
def to_plane(ans: str):
    a = normalize_answer(ans)
    
    #axial plane is also commonly called transverse
    if "axial" in a or "transverse" in a:
        return "axial"
    if "coronal" in a:
        return "coronal"
    if "sagittal" in a:
        return "sagittal"
    return None

#convert a raw answer into chest x-ray view labels
def to_view(ans: str):
    a = normalize_answer(ans)
    
    #PA view variants
    if a in {"pa", "p-a", "posteroanterior"} or "pa view" in a:
        return "pa"
    
    #AP view vairants
    if a in {"ap", "a-p", "anteroposterior"} or "ap view" in a:
        return "ap"
    
    #lateral view
    if "lateral" in a:
        return "lateral"
    return None

#convert a raw answer into course organ
def to_organ(ans: str):
    a = normalize_answer(ans)
    # brain/ neuro-related keywords
    if "brain" in a or "cerebell" in a or "ventricle" in a or "thalam" in a:
        return "brain"
    
    # chest-related keywords (lungs, heart, chest)
    if "lung" in a or "lungs" in a or "chest" in a or "heart" in a:
        return "chest"
    if "abdomen" in a or "pelvis" in a or "colon" in a or "bowel" in a:
        return "abdomen"
    
    #specific organs
    if "kidney" in a:
        return "kidney"
    if "liver" in a:
        return "liver"
    
    #Spine-related keywords
    if "spine" in a or "vertebra" in a:
        return "spine"
    
    return None

#convert a raw answer into MRI sequence labels
def to_sequence(ans: str):
    a = normalize_answer(ans)
    
    # MRI sequences / maps
    if "flair" in a:
        return "flair"
    if "t1" in a:
        return "t1"
    if "t2" in a:
        return "t2"
    if "dwi" in a or "diffusion" in a:
        return "dwi"
    if "adc" in a:
        return "adc"
    if "contrast" in a or "gadolinium" in a:
        return "contrast"
    return None

#convert raw answers into side labels
def to_side(ans: str):
    a = normalize_answer(ans)
    if "right" in a:
        return "right"
    if "left" in a:
        return "left"
    if "bilateral" in a or "both" in a:
        return "bilateral"
    return None

#task-aware answer mapping function
def map_answer(task: str, raw_answer: str):
    #closed yes/no style tasks (including pathologies and binary checks)
    if task in {"yesno", "pneumonia", "effusion", "pneumothorax", "cardiomegaly", "mass", "is_ct", "is_mri"}:
        return to_yesno(raw_answer)
    
    #open/multi-class tasks
    if task == "modality":
        return to_modality(raw_answer)
    if task == "plane":
        return to_plane(raw_answer)
    if task == "view":
        return to_view(raw_answer)
    if task == "organ":
        return to_organ(raw_answer)
    if task == "sequence":
        return to_sequence(raw_answer)
    if task == "side":
        return to_side(raw_answer)
    
    return None
