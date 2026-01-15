from datasets import load_dataset
import hashlib

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

ds = load_dataset("flaviagiammarino/vqa-rad")
train = ds["train"]
test = ds["test"]

print("Original sizes:")
print("train:", len(train))
print("test :", len(test))

# 1) image hash for duplicates (train only)
# vqa-rad usually has image as bytes in features. Common keys: "image" (PIL) or dict.
# We'll handle both safely.
def get_image_bytes(ex):
    img = ex["image"]
    # If it's already bytes-like (some datasets store dict with 'bytes')
    if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
        return img["bytes"]
    # If it's PIL Image
    try:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

train_hashes = []
for ex in train:
    b = get_image_bytes(ex)
    train_hashes.append(md5_bytes(b) if b is not None else None)

seen = set()
keep_idx = []
removed_dupes = 0
for i, h in enumerate(train_hashes):
    if h is None:
        keep_idx.append(i)
        continue
    if h in seen:
        removed_dupes += 1
    else:
        seen.add(h)
        keep_idx.append(i)

train_dedup = train.select(keep_idx)
print("\nAfter train image dedup:")
print("train_dedup:", len(train_dedup))
print("removed_dupes:", removed_dupes)

# 2) remove exact overlaps (image_hash + question + answer) between train and test
def make_key(ex, img_hash):
    q = norm_text(ex.get("question", ""))
    a = norm_text(ex.get("answer", ""))
    return (img_hash, q, a)

# Build test keys
test_hashes = []
for ex in test:
    b = get_image_bytes(ex)
    test_hashes.append(md5_bytes(b) if b is not None else None)

test_keys = set()
for ex, h in zip(test, test_hashes):
    test_keys.add(make_key(ex, h))

# Filter train_dedup
train_dedup_hashes = []
for ex in train_dedup:
    b = get_image_bytes(ex)
    train_dedup_hashes.append(md5_bytes(b) if b is not None else None)

keep_idx2 = []
removed_overlap = 0
for i, (ex, h) in enumerate(zip(train_dedup, train_dedup_hashes)):
    if make_key(ex, h) in test_keys:
        removed_overlap += 1
    else:
        keep_idx2.append(i)

train_clean = train_dedup.select(keep_idx2)
print("\nAfter removing train-test exact overlaps:")
print("train_clean:", len(train_clean))
print("removed_overlap:", removed_overlap)

# 3) validation split (10%) with seed 42
split = train_clean.train_test_split(test_size=0.1, seed=42)
train_final = split["train"]
val_final = split["test"]

print("\nFinal splits:")
print("train_final:", len(train_final))
print("val_final  :", len(val_final))
print("test_final :", len(test))
