import re

CLOSED_STARTS = ("is", "are", "was", "were", "do", "does", "did",
                 "can", "could", "has", "have", "had")

def normalize_answer(a: str) -> str:
    a = str(a).strip().lower()
    a = re.sub(r"\s+", " ", a)

    a = a.replace("xray", "x-ray")
    a = a.replace("x ray", "x-ray")
    a = a.replace("chest x ray", "chest x-ray")
    a = a.replace("ct scan", "ct")
    a = a.replace("computed tomography", "ct")
    a = a.replace("magnetic resonance imaging", "mri")
    return a

def is_closed_question(q: str) -> int:
    q = str(q).strip().lower()
    return int(q.startswith(CLOSED_STARTS))
