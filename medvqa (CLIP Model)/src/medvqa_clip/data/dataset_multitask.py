from torch.utils.data import Dataset
from medvqa_clip.data.router import route_question
from medvqa_clip.data.label_map import map_answer

#multitask dataset from MED-QVA
class MultiTaskVQADataset(Dataset):
    def __init__(self, hf_split, vocabs: dict):
        #HuggingFace dataset split
        self.data = hf_split
        
        #vocab bundle
        self.vocabs = vocabs

        # Precompute valid indices so training only uses examples
        self.indices = []
        
        #loop through full split and keep only valid
        for i in range(len(self.data)):
            ex = self.data[i]
            
            #read and normalize question + answer
            q = str(ex.get("question","")).strip()
            a = str(ex.get("answer",""))
            
            #route question into a task spec
            spec = route_question(q)
            
            #decide task name (whether ended/opened)
            task = spec.name if spec.name in vocabs else "yesno" if spec.kind == "closed" else "open"
            
            #Map raw dataset answer to normalize label
            mapped = map_answer(task, a)
            
            if mapped is None:
                continue
            if mapped not in vocabs[task]["stoi"]:
                continue
            
            #Keep this example index as valid training data
            self.indices.append(i)

    def __len__(self):
        #number of valid examples after filtering
        return len(self.indices)

    def __getitem__(self, idx: int):
        #Fetch the real dataset index from filtering indices last
        ex = self.data[self.indices[idx]]
        
        #Read raw fields
        image = ex["image"]
        question = str(ex.get("question","")).strip()
        answer = str(ex.get("answer",""))
        
        #route question again to determine task
        spec = route_question(question)
        task = spec.name if spec.name in self.vocabs else "yesno" if spec.kind == "closed" else "open"

        #route question again to clean label string
        mapped = map_answer(task, answer)
        
        #convert mapped label string
        label = self.vocabs[task]["stoi"][mapped]

        #return dict used by collate function
        return {
            "image": image,
            "question": question,
            "task": task,
            "label": label,
            "answer_text": mapped,
        }
