import torch
from torch.utils.data import DataLoader

from medvqa_clip.data.load_dataset import load_hf_dataset
from medvqa_clip.data.dataset_multitask import MultiTaskVQADataset
from medvqa_clip.data.collate_clip import CLIPMultiTaskCollate

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def prepare_loaders(dataset_name: str, vocabs: dict, clip_name: str, max_len: int,
                    batch_size: int, num_workers: int, seed: int, val_ratio: float):
    ds = load_hf_dataset(dataset_name, clean=True)
    train_raw = ds["train"] if "train" in ds else list(ds.values())[0]

    split = train_raw.train_test_split(test_size=val_ratio, seed=seed)
    train_split, val_split = split["train"], split["test"]

    train_set = MultiTaskVQADataset(train_split, vocabs=vocabs)
    val_set   = MultiTaskVQADataset(val_split, vocabs=vocabs)

    collate_fn = CLIPMultiTaskCollate(clip_name=clip_name, max_len=max_len)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader

def prepare_test_loader(dataset_name: str, vocabs: dict, clip_name: str, max_len: int,
                        batch_size: int, num_workers: int):
    ds = load_hf_dataset(dataset_name, clean=True)
    test_raw = ds["test"] if "test" in ds else list(ds.values())[-1]
    test_set = MultiTaskVQADataset(test_raw, vocabs=vocabs)
    collate_fn = CLIPMultiTaskCollate(clip_name=clip_name, max_len=max_len)

    return DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
