from ultralytics import YOLO
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import torch, gc

# --- CONFIG ---
MODEL_PATH = "tag_classifier.pt"
IMG_DIR = Path("../data")
TAG_CLASS_NAME = "tag"
CONF_THRESH = 0.8
PREFIX = "tag_"
DEVICE = "cuda:0"

# ----------------

model = YOLO(MODEL_PATH)
model.to(DEVICE)

images = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")]
counts = Counter()
renamed = []

for img_path in tqdm(images, desc="Classifying & renaming", unit="img"):
    # predict one image
    res = model.predict(
        source=str(img_path), stream=False, verbose=False, device=DEVICE
    )
    r = res[0]
    cid = r.probs.top1
    prob = float(r.probs.top1conf)
    cname = model.names[cid]

    if CONF_THRESH is None or prob >= CONF_THRESH:
        counts[cid] += 1

        if cname == TAG_CLASS_NAME and not img_path.name.startswith(PREFIX):
            new_path = img_path.with_name(f"{PREFIX}{img_path.name}")
            if new_path.exists():
                i = 1
                while (
                    cand := img_path.with_name(f"{PREFIX}{i}_{img_path.name}")
                ).exists():
                    i += 1
                new_path = cand
            img_path.rename(new_path)
            renamed.append((img_path.name, new_path.name))

    # free memory just in case
    torch.cuda.empty_cache() if DEVICE.startswith("cuda") else None
    gc.collect()

# --- Summary ---
print("\nCounts:")
for cid, n in counts.items():
    print(f"{model.names[cid]} ({cid}): {n}")

print("\nRenamed files:")
for old, new in renamed:
    print(f"{old} -> {new}")
