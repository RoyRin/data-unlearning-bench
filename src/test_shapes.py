from collections import Counter
from pathlib import Path
import torch
from paths import DATA_DIR

# ------------------------------------------------------------------
# Config – adjust to match how you store the tensors
# ------------------------------------------------------------------
root       = Path(DATA_DIR) / "living17"
files      = ["raw_tensors_tr_new.pt", "raw_tensors_val_new.pt"]  # training + val

# ------------------------------------------------------------------
# Scan every image once and record its spatial size
# ------------------------------------------------------------------
shape_counter = Counter()

for f in files:
    imgs, _ = torch.load(root / f)             # imgs: (N, 3, H, W) uint8 tensor
    for img in imgs:
        shape_counter[tuple(img.shape)] += 1   # (C,H,W) tuple acts as key

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("Unique image shapes found (shape → count):")
for shape, cnt in shape_counter.items():
    print(f"{shape}: {cnt}")

if len(shape_counter) == 1:
    print("\n✅  All images have the same size.")
else:
    print(f"\n⚠️  Mismatch detected: {len(shape_counter)} different shapes.")

from paths import check_hf_registry

oracle = check_hf_registry({"dataset": "living17", "model": "resnet18", "forget_id": 1, "N": 2}, mode="oracle_margins")
import pdb; pdb.set_trace()