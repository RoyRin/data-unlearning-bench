from pathlib import Path
import numpy as np
import sys
 

import torch 
from pathlib import Path
import numpy as np
import torch
from unlearning_bench.models import ResNet9
from unlearning_bench.datasets import get_cifar_dataloader
from unlearning_bench import train
from unlearning_bench import ORACLES_PATH


from unlearning.auditors.utils import (
    loader_factory,
    load_forget_set_indices,
)
from torchvision import models as torchvision_models


DATASET = "CIFAR10"



def train_oracles(
    forget_set_id,
    n_models: int,
    ckpt_dir: str,
    should_save_train_logits: bool = True,
    should_save_val_logits: bool = True,
    model_id_offset: int = 0,
    epochs=24,
    idx_start =0,
    evaluate=True,
    resnet18=False,
    ds_name = DATASET,
):
    """
    - model_id_offset is the offset to add to the model_id when saving the
    model, here only for a hacky use, feel free to ignore
    """
    forget_set_indices = load_forget_set_indices(ds_name,
                                                 forget_set_id)
    drop_last = False 
    shuffle=  True 
    # no shuffling, no augmentation
    if ds_name == 'QNLI':
        train_indices = np.arange(10_000)
    elif ds_name == 'LIVING17':
        train_indices = np.arange(44_200)
    else:
        train_indices = np.arange(50_000)

    #train_indices remove the forget_set_indices
    train_indices = np.setdiff1d(train_indices, forget_set_indices)
    print(f"size of train_indices: {len(train_indices)}")
    print(f"size of forget_set_indices: {len(forget_set_indices)}")

    print(f"computing from {n_models}")
    for model_id in range(n_models):
        print(f"training - {model_id}")
        print(f"ckpt_dir-{ckpt_dir}")
        if (ckpt_dir / f"val_logits_{model_id+ model_id_offset}.pt").exists():
            print(
                f"skipping model {model_id +model_id_offset} because logits already exist"
            )
            continue
        
        train_loader = loader_factory(ds_name, indices=train_indices, shuffle= shuffle, indexed=True, drop_last = drop_last)

        eval_loader = loader_factory(ds_name, split="val", shuffle= shuffle, indexed=True)


        if resnet18:
            print(f"training resnet18!")
            print(f"ckpt_dir- {ckpt_dir}")
            from unlearning.models.resnet9 import WrappedModel

            model = torchvision_models.resnet18(num_classes=10)
            # model = ResNet9(num_classes=10)
            model = WrappedModel(model).cuda()

        else:
            model = ResNet9(num_classes=10, wrapped=True).cuda()

        # checkpoint_epochs # every 5 epochs
        checkpoint_epochs = list(range(5, epochs, 5)) + [epochs - 1]
        checkpoint_epochs = [10, 15, epochs-1]
        
        model = train.train_cifar10(
            model=model,
            loader=train_loader,
            checkpoints_dir=ckpt_dir,
            model_id=model_id + model_id_offset,
            checkpoint_epochs=checkpoint_epochs,
            epochs=epochs,
            eval_loader=None,
            should_save_logits=should_save_train_logits,
        )
        # eval model
        if evaluate:
            acc = train.eval_cifar10(model, eval_loader)
            print(f"eval acc-  {acc}")
            print("------")

        if should_save_train_logits:
            full_dataloader_unshuffled = get_cifar_dataloader(
                num_workers=2, indexed=True
            )
            logits = train.get_logits(model, full_dataloader_unshuffled)
            logits_path = ckpt_dir / f"train_logits_{model_id + model_id_offset}.pt"
            torch.save(logits, logits_path)
            print(f"saved logits to {logits_path}")
        if should_save_val_logits:
            val_dataloader_unshuffled = get_cifar_dataloader(
                split="val", num_workers=2, indexed=True
            )
            val_logits = train.get_logits(model, val_dataloader_unshuffled)
            val_logits_path = ckpt_dir / f"val_logits_{model_id + model_id_offset}.pt"
            torch.save(val_logits, val_logits_path)
            print(f"saved logits to {val_logits_path}")
    return 0  # return 0 if everything went well



if __name__ == "__main__":
    import sys
    from pathlib import Path

    try:
        job_id = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID: 0 to 49
    except:
        job_id = 0

    N_forget_sets = 10
    N_machines_per_fs = 5
    N_models_total = 200
    N_models_per_job = N_models_total // N_machines_per_fs

    forget_set_ind = ( job_id // N_machines_per_fs)+1  # 0 to 9
    machine_ind = job_id % N_machines_per_fs      # 0 to 4
    model_id_offset = machine_ind * N_models_per_job

    ds_name = "CIFAR10"
    MODELS_PATH = ORACLES_PATH
    MODELS_PATH = MODELS_PATH / ds_name / f"forget_set_{forget_set_ind}"
    MODELS_PATH.mkdir(exist_ok=True, parents=True)

    print(f"[Job {job_id}] forget_set_ind={forget_set_ind}, machine_ind={machine_ind}, model_id_offset={model_id_offset}")
    print(f"MODELS_PATH = {MODELS_PATH}")

    epochs = 24
    should_save_train_logits = True

    try:
        train_oracles(
            forget_set_id=forget_set_ind ,
            n_models=N_models_per_job,
            ckpt_dir=MODELS_PATH,
            should_save_train_logits=should_save_train_logits,
            should_save_val_logits=True,
            model_id_offset=model_id_offset,
            epochs=epochs,
        )
    except Exception as e:
        print(f"Exception occurred: {e}")
        
        
