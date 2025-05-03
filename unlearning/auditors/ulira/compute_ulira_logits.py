from trak.modelout_functions import AbstractModelOutput
from torch.nn import Module
from torch import Tensor
from typing import Iterable

import numpy as np
import torch as ch
from pathlib import Path
from matplotlib import pyplot as plt
from trak import TRAKer

from unlearning.datasets.cifar10 import get_cifar_dataloader

from unlearning import BASE_DIR, LOG_DIR, ULIRA_BASE_DIR

from unlearning.training.train import wrapper_for_train_cifar10_on_subset_submitit


class LogitModelOutput(AbstractModelOutput):

    def __init__(self, logit_ind: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.logit_ind = logit_ind

        self.softmax = ch.nn.Softmax(-1)
        self.loss_temperature = temperature

    def get_output(
        self,
        model: Module,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        image: Tensor,
        label: Tensor,
    ):
        logits = ch.func.functional_call(model, (weights, buffers),
                                         image.unsqueeze(0))
        return logits[0, self.logit_ind].sum()

    def get_out_to_loss_grad(
        self,
        model: Module,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        batch: Iterable[Tensor],
    ) -> Tensor:
        raise UserWarning(
            "This class should only be used for scoring, not for featurizing")


from importlib import reload
#from notebooks.utils import get_executor, submit_job

from notebooks import utils

reload(utils)

if __name__ == "__main__":

    # This script is for computing ULIRA models for cifar10

    # (should be made into its own script)
    import sys
    try:
        ind = int(sys.argv[1])
    except:
        ind = 0

    ds_name = "CIFAR10"

    MODELS_PATH = BASE_DIR / "resnet_long/"
    MASKS_PATH = BASE_DIR / "training_masks.npy"

    # MODELS_PATH.mkdir(exist_ok=True, parents=True)

    N_models_per_job = 10
    epochs = 100

    MODELS_PATH = BASE_DIR / f"resnet_long_{epochs}/"

    batch_args = []

    should_save_train_logits = True  # False

    model_id_offset = 0
    idx_start = ind * N_models_per_job
    trials = 5
    resnet18 = True

    for i in range(trials):
        try:
            wrapper_for_train_cifar10_on_subset_submitit(
                masks_path=MASKS_PATH,
                idx_start=idx_start,
                n_models=N_models_per_job,
                ckpt_dir=MODELS_PATH,
                should_save_train_logits=should_save_train_logits,
                should_save_val_logits=True,
                model_id_offset=model_id_offset,
                epochs=epochs,
                evaluate=True,
                resnet18=resnet18)

            break
        except Exception as e:
            print(f"Error in trial {i}: {e}")
            # wait 30 seconds
            import time
            time.sleep(30)
