# stdlib deps
import os

# project deps
from paths import CONFIG_DIR
from unlearning import UNLEARNING_METHODS
from datasets import DATASETS
from models import MODELS

# external deps
import yaml

def load_config(config_name):
    config_path = CONFIG_DIR / config_name
    assert os.path.exists(config_path), f"config {args.config_name} not in {CONFIG_DIR}"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_config(config):
    required_fields = {
            "unlearning_method": (lambda m: m in UNLEARNING_METHODS, f"unlearning_method not in {UNLEARNING_METHODS.keys()}"),
            "lr": (lambda lr: isinstance(lr, float) and lr>0, "lr should be a positive float"),
            "epochs": (lambda epochs: isinstance(epochs, int) and epochs > 0, "epochs should be a positive integer"),
            "dataset": (lambda d: d in DATASETS, f"dataset not in {DATASETS.keys()}"),
            "optimizer": (lambda o: o == "sgd", "optimizer not sgd"),
            "model": (lambda m: m == m in MODELS, f"model not in {MODELS.keys()}"),
            "N": (lambda epochs: isinstance(epochs, int) and epochs > 0, "N should be a positive integer"),
            "forget_id": (lambda epochs: isinstance(epochs, int) and epochs > 0, "forget_id should be a positive integer"),
    }
    missing_fields = [mf for mf in required_fields if mf not in config]
    assert len(missing_fields) == 0, f"config is missing required fields {missing_fields}"
    wrong_req_fields = [wrf for wrf in required_fields if not required_fields[wrf][0](config[wrf])]
    assert len(wrong_req_fields) == 0, f"wrong required fields {[(wrf, required_fields[wrf][1]) for wrf in wrong_req_fields]}"
