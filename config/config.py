# config.py
import logging.config
import sys
from pathlib import Path

import pretty_errors  # NOQA: F401 (imported but unused)
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_REGISTRY = Path(BASE_DIR, "mlruns")
STORES_DIR = Path(BASE_DIR,"stores")
BLOB_STORE = Path(STORES_DIR, "blob")
BLOB_STORE.mkdir(parents=True, exist_ok=True)

# create directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = (
    "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
)
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"

STOPWORDS = [
    "i",
    "me",
    "my",
    "won't",
    "wouldn",
    "wouldn't",
]

ACCEPTED_TAGS = ["natural-language-processing", "computer-vision", "mlops", "graph-learning"]

LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)  # pretty formatting

# Sample messages (note that we use configured `logger` now)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
