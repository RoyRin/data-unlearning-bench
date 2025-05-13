# test_resnet18_living17.py
import torch
from tqdm import tqdm
from paths import check_hf_registry

from datasets import DATASETS               # your registry

# ---------------------------------------------------------------------
# config ­– tweak these three lines if needed
# ---------------------------------------------------------------------
BATCH_SIZE = 256
NUM_CLASSES = 17                            # Living-17
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = check_hf_registry({
    "dataset": "living17",
    "model": "resnet18",
    "N": 1,
}, "pretrain_checkpoints")[0]
# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
val_loader = DATASETS["living17_new"]["loader"](
    split="val",
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------
model = model.to(DEVICE).eval()

# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------
correct, total = 0, 0
with torch.no_grad():
    for x, y in tqdm(val_loader, desc="validating"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast("cuda"):
            logits = model(x)
        preds  = logits.argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)

top1 = correct / total * 100
print(f"ResNet-18 | Living-17 val accuracy: {top1:.2f}%")
