# config.py
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# create directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = (
    "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
)
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
