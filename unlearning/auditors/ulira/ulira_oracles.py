from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (RandomHorizontalFlip, Cutout, RandomTranslate,
                             Convert, ToDevice, ToTensor, ToTorchImage)

from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

import os
import numba

numba.set_num_threads(1)
from memorization import utils
from memorization import cifar10
import sys
import datetime
import signal
import sys

from unlearning.training.train import train_cifar10


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


NUM_CLASSES = 10

if __name__ == "__main__":
    overwrite = False
    beginning_time = datetime.datetime.now()

    data_dir = utils.get_data_dir()

    index = int(sys.argv[1])

    plans = np.load(data_dir / "plans.npy")
    plan = plans[index]
    print(f"loaded plan {index}")

    print(plan[:100])
    print(f"length - {len(plan)}")
    ####
    model_save_dir = data_dir / "models" / f"model_{index}"
    # save
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_save_dir / "model.pth"

    # check if model exists
    if save_memorization_preds_path.exists() and not overwrite:
        print(f"model exists, skipping")
        sys.exit(0)

    full_dataset = torchvision.datasets.CIFAR10('/tmp',
                                                train=True,
                                                download=True)

    full_targets = np.array(full_dataset.targets)

    subset = ch.utils.data.Subset(full_dataset, plan)

    datasets = {
        'train': subset,
        'test': torchvision.datasets.CIFAR10('/tmp',
                                             train=False,
                                             download=True),
    }

    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'/tmp/cifar_{name}__{index}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    BATCH_SIZE = 512

    loaders = {}
    for name in ['train']:  #, "train_full", 'test']:

        device = ch.device('cuda:0')  # Create a device object

        # Modify your label and image pipelines:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(), ToDevice(device),
            Squeeze()
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Then, proceed with the rest of the pipeline
        if 'train' in name:
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(
                    int,
                    CIFAR_MEAN))),  # Note Cutout is done before normalization.
            ])

        image_pipeline.extend([
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        loaders[name] = Loader(f'/tmp/cifar_{name}.beton',
                               batch_size=BATCH_SIZE,
                               num_workers=1,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                                   'label': label_pipeline
                               })

    #####
    # setting up training
    #####

    EPOCHS = 25
    #train_cifar10()
    print(f"starting training")

    opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = 50000 // BATCH_SIZE
    lr_schedule = np.interp(np.arange((EPOCHS + 1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    print(f"starting!")
    from tqdm import tqdm

    all_preds = []

    start_time = datetime.datetime.now()

    #####
    # Training
    #####

    for ep in range(EPOCHS):
        print(f"Epoch {ep}")

        if ep in epochs_of_interest:
            print(f"making inferences")
            all_preds.append(get_predictions(model, loaders['train_full']))
            model.train()

        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    # get targets for train_full
    #targets = np.array(datasets['train_full'].targets)

    #targets = np.array([lab for _, lab in loaders['train_full']])
    all_preds = np.array(all_preds)

    memorization_preds = {
        "all_predictions": all_preds,
        "targets": full_targets,
        "epochs_of_interest": epochs_of_interest
    }
    # save preds

    np.savez(save_memorization_preds_path, **memorization_preds)

    end_time = datetime.datetime.now()
    print(f"total_time: {end_time - start_time}")

    # save the model
    ch.save(model.state_dict(), model_save_path)  #'model.pth')
    # save predictions

    np.save(predictions_save_path, all_preds)

    print(f"saved to {model_save_dir}")
    model.eval()

    #####
    # Eval
    #####

    print(f"eval")

    with ch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in tqdm(loaders['test']):
            with autocast():
                out = (model(ims) +
                       model(ch.fliplr(ims))) / 2.  # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

        print(f'Accuracy: {total_correct / total_num * 100:.1f}%')

    print(f"done!")
    end_time = datetime.datetime.now()

    signal.signal(signal.SIGINT, signal_handler)

    #sys.exit(0)

    print(f"total_time: {end_time - beginning_time}")
    os._exit(0)  # Use as a last resort
    print(f"total_time: {end_time - beginning_time}")

    sys.exit(0)
