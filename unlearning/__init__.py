from pathlib import Path

dm_scores_path = Path(
    f"regression_v2/all_logit_vitaly_infls_denoised.npy")  # datamodels path

BASE_DIR = Path(
    "/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/")  # where precomputed models lie
ORACLE_BASE_DIR = Path(f"/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/oracles/"
                       )  # where oracles (retrained models) lie
if not BASE_DIR.exists():
    print(f"Was not able to find precomputed_models directory in {BASE_DIR}")
    print(f"... Continuing anyways. but proceed with caution!")

ULIRA_BASE_DIR = Path("unlearning/precomputed_models/ULIRA_clean/"
                      )  # where precomputed ULIRA models lie

MARGINS_OVER_TIME_RESULTS_DIR = Path(
    "unlearning/KLOM/margins_over_time"
)  # directory for storing margins over time

RESULTS_DIR = Path("PAPER_RESULTS/")

LOG_DIR = Path("catered_out/unlearning")
