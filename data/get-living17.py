from huggingface_hub import snapshot_download
from pathlib import Path

target_dir = Path("./living17")      # change to where you want the folder
target_dir.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="royrin/KLOM-models",            # full repo name
    repo_type="dataset",                     # because this is under /datasets/
    local_dir=target_dir,                    # download straight into this path
    allow_patterns=["data/living17/*"],      # only pull the desired sub-folder
    tqdm_class=None                          # optional: silence progress bars
)

print(f"Done! Files are in {target_dir.resolve()}")
