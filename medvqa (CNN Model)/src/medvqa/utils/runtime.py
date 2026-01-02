import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from medvqa.data.load_dataset import load_hf_dataset
from medvqa.data.dataset import MedVQADataset
from medvqa.data.collate import CNNVQACollate

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_closed_balanced_sampler(dataset: MedVQADataset):
    w = []
    cnt = {0: 0, 1: 0}
    for i in range(len(dataset)):
        ex = dataset[i]
        if ex["is_closed_q"] == 1 and ex["closed_label"] in (0, 1):
            cnt[int(ex["closed_label"])] += 1

    for i in range(len(dataset)):
        ex = dataset[i]
        if ex["is_closed_q"] == 1 and ex["closed_label"] in (0, 1):
            y = int(ex["closed_label"])
            w.append(1.0 / max(1, cnt[y]))
        else:
            w.append(1.0)

    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
    return sampler, cnt

def prepare_loaders(
    dataset_name: str,
    vocab: dict,
    text_model_name: str,
    img_size: int,
    max_len: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float = 0.1,
    balanced_closed_sampler: bool = False,
):
    ds = load_hf_dataset(dataset_name, clean=True)
    train_raw = ds["train"] if "train" in ds else list(ds.values())[0]

    split = train_raw.train_test_split(test_size=val_ratio, seed=seed)
    train_split = split["train"]
    val_split = split["test"]

    train_set = MedVQADataset(train_split, vocab)
    val_set = MedVQADataset(val_split, vocab)

    collate_fn = CNNVQACollate(text_model_name, img_size=img_size, max_len=max_len)

    sampler = None
    sampler_info = None
    if balanced_closed_sampler:
        sampler, cnt = _make_closed_balanced_sampler(train_set)
        sampler_info = cnt

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, sampler_info

def prepare_test_loader(dataset_name: str, vocab: dict, text_model_name: str, img_size: int, max_len: int, batch_size: int, num_workers: int):
    ds = load_hf_dataset(dataset_name, clean=True)
    test_raw = ds["test"] if "test" in ds else (ds["validation"] if "validation" in ds else ds["train"])
    test_set = MedVQADataset(test_raw, vocab)

    collate_fn = CNNVQACollate(text_model_name, img_size=img_size, max_len=max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return test_loader
