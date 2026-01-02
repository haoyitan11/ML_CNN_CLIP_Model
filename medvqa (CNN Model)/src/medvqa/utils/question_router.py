import re
from dataclasses import dataclass
from typing import Literal

QType = Literal["closed_yesno", "open"]
SubType = Literal["modality", "plane", "view", "organ", "laterality", "sequence", "structure", "generic"]

@dataclass
class Route:
    qtype: QType
    subtype: SubType
    debug: str

_RE_VIEW_EXPECT_OPEN = re.compile(r"\b(pa|ap)\b.*\bview\b", re.IGNORECASE)
_RE_PLANE_WORD = re.compile(r"\b(axial|coronal|sagittal)\b", re.IGNORECASE)
_RE_STARTS_CLOSED = re.compile(r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would)\b", re.IGNORECASE)

def route_question(q: str) -> Route:
    q0 = "" if q is None else str(q)
    ql = q0.strip().lower()

    if "imaging modality" in ql or "what imaging modality" in ql or "type of scan" in ql or "what scan" in ql:
        return Route("open", "modality", "matched modality keywords")

    if "mri sequence" in ql or ("sequence" in ql and "what" in ql):
        return Route("open", "sequence", "matched sequence keywords")

    if "plane" in ql or "orientation" in ql:
        return Route("open", "plane", "matched plane/orientation keywords")

    if "view" in ql:
        if _RE_VIEW_EXPECT_OPEN.search(q0):
            return Route("open", "view", "special-case: PA/AP view expected open")
        return Route("open", "view", "matched view keyword")

    if "which structure" in ql:
        return Route("open", "structure", "matched 'which structure'")

    if "organ" in ql or "body region" in ql or "body part" in ql or "structure" in ql:
        return Route("open", "organ", "matched organ/body keywords")

    if "right side" in ql or "left side" in ql or "on the right" in ql or "on the left" in ql:
        return Route("open", "laterality", "matched laterality keywords")

    starts_closed = bool(_RE_STARTS_CLOSED.match(ql))
    if starts_closed and _RE_PLANE_WORD.search(q0) and ("plane" in ql or "image" in ql):
        return Route("open", "plane", "special-case: plane word present; treat open")
    if starts_closed:
        return Route("closed_yesno", "generic", "prefix suggests yes/no")

    return Route("open", "generic", "default open")
