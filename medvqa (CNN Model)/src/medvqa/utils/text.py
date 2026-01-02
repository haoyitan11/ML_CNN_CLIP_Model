import re
from typing import Tuple

def normalize_answer(a: str) -> str:
    a = "" if a is None else str(a)
    a = a.strip().lower()
    a = re.sub(r"\s+", " ", a)
    a = a.replace("xray", "x-ray")
    a = a.replace("x ray", "x-ray")
    a = a.replace("chest x ray", "chest x-ray")
    a = a.replace("ct scan", "ct")
    a = a.replace("computed tomography", "ct")
    a = a.replace("magnetic resonance imaging", "mri")
    a = a.replace("mr flair", "flair")
    a = a.replace("mri flair", "flair")
    a = a.replace("t2 weighted", "t2")
    a = a.replace("t2-weighted", "t2")
    a = a.replace("adc map", "adc")
    return a

def yes_no_label(ans: str):
    ans = normalize_answer(ans)
    if ans == "yes":
        return 1
    if ans == "no":
        return 0
    return None

CLOSED_STARTS: Tuple[str, ...] = (
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "has", "have", "had", "will", "would"
)

def is_closed_question_prefix(q: str) -> int:
    q = "" if q is None else str(q)
    q = q.strip().lower()
    return int(q.startswith(CLOSED_STARTS))
