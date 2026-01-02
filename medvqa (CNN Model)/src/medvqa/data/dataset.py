from torch.utils.data import Dataset
from medvqa.utils.text import normalize_answer, yes_no_label, is_closed_question_prefix

class MedVQADataset(Dataset):
    def __init__(self, hf_split, vocab: dict):
        #HuggingFace split (train/val/test)
        self.data = hf_split
        
        #Answer > id
        self.stoi = vocab["stoi"]
        #id > answer
        self.itos = vocab["itos"]
        self.other_id = self.stoi.get("OTHER", len(self.itos) - 1)

    def __len__(self):
        #number of samples
        return len(self.data)

    def __getitem__(self, idx: int):
        #get one example
        ex = self.data[idx]
        #PIL image
        image = ex["image"]
        question = str(ex.get("question", "")).strip()

        #normalize answer text
        ans_norm = normalize_answer(ex.get("answer", ""))
        #open-answer class id
        open_label = self.stoi.get(ans_norm, self.other_id)

        #try map to yes/no
        yn = yes_no_label(ans_norm)
        # -1 means not yes/no
        closed_label = -1 if yn is None else int(yn)

        #detect closed question by prefix
        is_closed_q = is_closed_question_prefix(question)

        return {
            "image": image,
            "question": question,
            "open_label": open_label,
            "closed_label": closed_label,
            "is_closed_q": is_closed_q,
            "answer_text": ans_norm,
        }
