# stdlib deps
import os
import itertools

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
            "epochs": (lambda epochs: isinstance(epochs, list) and all(isinstance(ll, int) and ll > 0 for ll in epochs) and all(a <= b for a, b in zip(epochs, epochs[1:])), "epochs should be a sorted list of positive integers"),
            "dataset": (lambda d: d in DATASETS, f"dataset not in {DATASETS.keys()}"),
            "optimizer": (lambda o: o == "sgd", "optimizer not sgd"),
            "model": (lambda m: m == m in MODELS, f"model not in {MODELS.keys()}"),
            "N": (lambda epochs: isinstance(epochs, int) and epochs > 0, "N should be a positive integer"),
            "forget_id": (lambda epochs: isinstance(epochs, int) and epochs > 0, "forget_id should be a positive integer"),
    }
    missing_fields = [mf for mf in required_fields if mf not in config]
    assert len(missing_fields) == 0, f"config is missing required fields {missing_fields}"
    try:
        wrong_req_fields = [wrf for wrf in required_fields if not required_fields[wrf][0](config[wrf])]
    except:
        import pdb; pdb.set_trace()
    assert len(wrong_req_fields) == 0, f"wrong required fields {[(wrf, required_fields[wrf][1]) for wrf in wrong_req_fields]}"

def get_config_name(config, list_keys = None) -> str:
    list_keys = set(list_keys or [])
    parts = []
    for k in sorted(config):
        v = config[k]
        if k in list_keys and isinstance(v, list):
            v = "[" + ",".join(map(str, v)) + "]"
        parts.append(f"{k}-{v}")
    return "_".join(parts) + ".yml"

def generate_configs(configs_dict, configs_dir, list_keys):
    list_keys = set(list_keys or [])
    os.makedirs(configs_dir, exist_ok=True)
    prod_keys = [k for k in configs_dict if k not in list_keys]
    counter = 0
    for values in itertools.product(*(configs_dict[k] for k in prod_keys)):
        cfg = {k: v for k, v in zip(prod_keys, values)}
        cfg.update({k: configs_dict[k] for k in list_keys})
        check_config(cfg)
        config_name = get_config_name(cfg, list_keys)
        with open(os.path.join(configs_dir, config_name), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        counter+=1
    print(f"Generated {counter} configs")

if __name__=="__main__":
    # from C3 https://arxiv.org/pdf/2410.23232
    # Gradient Ascent: Optimized with SGD. Learning rates: {1e-5, 1e-3, 1e-2}. Epochs: {1, 3, 5, 7, 10}
    default_params = {
            "optimizer": ["sgd"],
            "model": ["resnet9"],
            "N": [100],
            "forget_id": [3,5],
            "batch_size": [64],
            "dataset": ["cifar10"]
    }
    ascent_configs = {
            "unlearning_method": ["ascent_forget"],
            "lr": [1e-5, 1e-3, 1e-2],
            "epochs": [1, 3, 5, 7, 10],
            **default_params
    }
    do_nothing_configs = {
            "unlearning_method": ["do_nothing"],
            "lr": [10000.0],
            "epochs": [1],
            **default_params
    }
    scrub_configs = {
            "unlearning_method": ["scrub"],
            "lr": [5e-3, 1e-3, 5e-4, 5e-5],
            "epochs": [5, 7, 10],
            "ascent_epochs": [3, 5],
            "forget_batch_size": [32, 64],
            **default_params
    }
    # generate_configs(ascent_configs, CONFIG_DIR, ["epochs"])
    # generate_configs(do_nothing_configs, CONFIG_DIR, ["epochs"])
    generate_configs(scrub_configs, CONFIG_DIR, ["epochs"])
