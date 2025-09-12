import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def to_np(filename, np_dir, pt_dir):
    np_path = np_dir / Path(str(filename).replace(".pt", ".npy"))
    pt_path = pt_dir / filename
    data = torch.load(pt_path, map_location="cpu")
    arr = data["margins"].numpy()
    np.save(np_path, arr)

var_np_dir_parent = Path("np_margins")
var_pt_dir_parent = Path("nanogpt_margins")
for mm in ["full_model_margins", "oracle_margins"]:
    var_np_dir = var_np_dir_parent / mm 
    var_pt_dir = var_pt_dir_parent / mm
    for vf in tqdm(var_pt_dir.iterdir()):
        to_np(vf.name, var_np_dir, var_pt_dir)
