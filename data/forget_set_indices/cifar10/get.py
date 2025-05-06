import requests
import numpy as np
import io

contents = [
    np.load(
        io.BytesIO(
            requests.get(
                f"https://huggingface.co/datasets/royrin/KLOM-models/resolve/main/"
                f"forget_set_indices/CIFAR10/forget_set_{i}.npy",
                timeout=30
            ).content
        ),
        allow_pickle=False    # or True if you really need pickles
    )
    for i in range(1, 10)
]
for ii, cc in enumerate(contents):
    np.save(f"forget_set_{ii}.npy", cc)
