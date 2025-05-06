# stdlib deps
from pathlib import Path

# Paths with respect to files inside the repo src/ folder
SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DATA_DIR = REPO_DIR / "data"
EVAL_DIR = DATA_DIR / "eval"
MARGINS_DIR = DATA_DIR / "margins"
FORGET_INDICES_DIR = DATA_DIR / "forget_set_indices"
ORACLES_DIR = DATA_DIR / "oracles"
