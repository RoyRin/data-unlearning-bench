from pathlib import Path

dm_scores_path = Path(
    f"regression_v2/all_logit_vitaly_infls_denoised.npy")  # datamodels path

BASE_DIR = Path(
    "/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/")  # where precomputed models lie
ORACLE_BASE_DIR = BASE_DIR / "oracle_models/"

# where oracles (retrained models) lie
if not BASE_DIR.exists():
    print(f"Was not able to find precomputed_models directory in {BASE_DIR}")
    print(f"... Continuing anyways. but proceed with caution!")

CIFAR_ORACLE_MODELS = ORACLE_BASE_DIR / "CIFAR10"
LIVING17_ORACLE_MODELS = ORACLE_BASE_DIR / "LIVING17"
LIVING17_FULL_MODELS = Path("/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/full_models/LIVING17/")
# where precomputed ULIRA models lie
ULIRA_BASE_DIR = Path("unlearning/precomputed_models/ULIRA_clean/")  

MARGINS_OVER_TIME_RESULTS_DIR = Path(
    "unlearning/KLOM/margins_over_time"
)  # directory for storing margins over time

RESULTS_DIR = Path("PAPER_RESULTS/")


LIVING17_ROOT = BASE_DIR  / "LIVING17_dataset/living17"


LOG_DIR = Path("catered_out/unlearning")