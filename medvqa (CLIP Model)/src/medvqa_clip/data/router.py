from dataclasses import dataclass

@dataclass
class RouteSpec:
    name: str   # task name
    kind: str   # 'closed' or 'open' or 'task'
    notes: str = ""

def route_question(q: str) -> RouteSpec:
    #normalize input question
    ql = str(q).strip().lower()

    #modality question
    if "imaging modality" in ql or "modality" in ql or "what scan" in ql:
        return RouteSpec(name="modality", kind="open", notes="modality")

    #Explicit CT/MRI binary question
    if "is this a ct" in ql or "ct scan" in ql:
        return RouteSpec(name="is_ct", kind="closed")
    if "is this an mri" in ql:
        return RouteSpec(name="is_mri", kind="closed")

    #Plane questions
    if "what plane" in ql or "which plane" in ql or "plane is this" in ql:
        return RouteSpec(name="plane", kind="open", notes="plane")

    #Chest x-ray view questions
    if "pa view" in ql or "ap view" in ql or "view is this" in ql or "pa or ap" in ql:
        return RouteSpec(name="view", kind="open", notes="view")

    # organ / region question
    if "what organ" in ql or "body region" in ql or "what body region" in ql:
        return RouteSpec(name="organ", kind="open", notes="organ-region")

    # MRI sequence question 
    if "mri sequence" in ql or "sequence is shown" in ql or "adc map" in ql or "dwi" in ql or "flair" in ql or "t1" in ql or "t2" in ql:
        return RouteSpec(name="sequence", kind="open", notes="mri-seq")

    # side
    if "right side" in ql:
        return RouteSpec(name="side", kind="closed", notes="right")
    if "left side" in ql:
        return RouteSpec(name="side", kind="closed", notes="left")

    # chest pathology findings (closed yes/no)
    if "pneumonia" in ql:
        return RouteSpec(name="pneumonia", kind="closed")
    if "pleural effusion" in ql or "effusion" in ql:
        return RouteSpec(name="effusion", kind="closed")
    if "pneumothorax" in ql:
        return RouteSpec(name="pneumothorax", kind="closed")
    if "cardiomegaly" in ql:
        return RouteSpec(name="cardiomegaly", kind="closed")
    if "mass" in ql:
        return RouteSpec(name="mass", kind="closed")

    # generic yes/no
    if ql.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ")):
        return RouteSpec(name="yesno", kind="closed", notes="generic")

    return RouteSpec(name="open", kind="open", notes="fallback")
