from pathlib import Path

__CURR_DIR__ = Path(__file__).resolve().parent

ROOT_PROJECT_DIR = (__CURR_DIR__ / '../').absolute().resolve()

SRC_DIR = ROOT_PROJECT_DIR / 'src'
CHECKPOINTS_DIR = SRC_DIR / 'checkpoints'
DATA_DIR = SRC_DIR / "data"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)